import math
from os.path import exists, join

import gym
import gym.envs.box2d
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms

from .models_of_world_model import VAE, Controller, MDRNNCell
from .models_of_world_model.mdrnn import gmm_loss
# # A bit dirty: manually change size of car racing env
# gym.envs.box2d.car_racing.STATE_W, gym.envs.box2d.car_racing.STATE_H = 64, 64

# Hardcoded for now
ASIZE, LSIZE, RSIZE, RED_SIZE, SIZE =\
    2, 32, 256, 96, 96

# Same
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((RED_SIZE, RED_SIZE)),
    transforms.ToTensor()
])


def process_state_dict(state_dict: dict):
    key_list = list(state_dict.keys())
    for k in key_list:
        if k.endswith('_l0'):
            new_k = k.rstrip('_l0')
            state_dict[new_k] = state_dict[k].clone()
            state_dict.pop(k)
    return state_dict


class World_model(nn.Module):
    """ Utility to generate rollouts.

    Encapsulate everything that is needed to generate rollouts in the TRUE ENV
    using a controller with previously trained VAE and MDRNN.

    :attr vae: VAE model loaded from mdir/vae
    :attr mdrnn: MDRNN model loaded from mdir/mdrnn
    :attr controller: Controller, either loaded from mdir/ctrl or randomly
        initialized
    """

    def __init__(self, device):
        super(World_model, self).__init__()
        """ Build vae, rnn, controller and environment. """
        # Loading world model and vae
        vae_file, rnn_file = \
            [join('exp_dir/', m, 'best.tar')
             for m in ['vae', 'mdrnn']]

        # assert exists(vae_file) and exists(rnn_file),\
        #     "Either vae or mdrnn is untrained."

        vae_state = torch.load(vae_file)
        self.vae = VAE(4, LSIZE)
        self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5)
        if exists(rnn_file):
            rnn_state = torch.load(rnn_file)
            state_dict = process_state_dict(rnn_state['state_dict'])
            self.mdrnn.load_state_dict(state_dict)

        self.device = device
        self.actor_critic = ActorCriticWorldModel()
        self.hidden = torch.zeros(12, RSIZE*2).to(self.device)
        self.Loss = {'VAE_Loss': 0, 'MDN_Loss': 0}

    @torch.no_grad()
    def world_model(self, obs):
        reconsturct_x, self.latent_mu, log_var = self.vae(obs/255.0, no_decoder=True)
        self.processed_obs = torch.cat(
            (self.latent_mu, self.hidden[:, :RSIZE]), dim=1).detach()
        return self.processed_obs

    @torch.no_grad()
    def update_hidden(self, actions):
        mus, sigmas, logpi, _, _, next_hidden = self.mdrnn(
            actions, self.latent_mu, [self.hidden[:, :RSIZE], self.hidden[:, RSIZE:]], with_MDN=False)
        self.hidden = torch.cat(next_hidden, dim=1)

    def get_action_and_transition(self, obs, next_hidden=None, next_obs=None):
        """ Get action and transition.
        Encode obs to latent using the VAE, then obtain estimation for next
        latent and next hidden state using the MDRNN and compute the controller
        corresponding action.
        :args obs: current observation (1 x 3 x 64 x 64) torch tensor
        :args hidden: current hidden state (1 x 256) torch tensor
        :returns: (action, next_hidden)
            - action: 1D np array
            - next_hidden (1 x 256) torch tensor
        """
        loss = False
        if next_hidden is not None and next_obs is not None:
            self.hidden = next_hidden
            loss = True

        reconsturct_x, latent_mu, log_var = self.vae(obs/255.0)
        if loss:
            self.Loss['VAE_Loss'] = self.VAE_loss(
                reconsturct_x, obs/255.0, latent_mu, log_var)

        self.processed_obs = torch.cat(
            (latent_mu, self.hidden[:, :RSIZE]), dim=1).detach()
        logits, actor_logstd, value = self.actor_critic(self.processed_obs)
        actions, action_log_probs = self.compute_action(logits, actor_logstd)

        mus, sigmas, logpi, _, _, next_hidden = self.mdrnn(
            actions, latent_mu, [self.hidden[:, :RSIZE], self.hidden[:, RSIZE:]])

        if loss:
            with torch.no_grad():
                next_obs_mu, next_obs_logsigma = self.vae(next_obs/255.0)[1:]
                next_latent = self.get_latent(next_obs_mu, next_obs_logsigma)
            self.Loss['MDN_Loss'] = gmm_loss(
                next_latent, mus, sigmas, logpi)

        self.hidden = torch.cat(next_hidden, dim=1)
        return actions, value, action_log_probs

    def forward(self, obs, next_hidden=None, next_obs=None):

        # action, value, action_log_probs = self.get_action_and_transition(
        #     obs, next_hidden, next_obs)
        logits, actor_logstd, value = self.actor_critic(obs)
        return logits, actor_logstd, value

    def compute_action(self, means, log_std):
        normal_distribution = torch.distributions.normal.Normal(
            means, torch.exp(log_std))
        actions = normal_distribution.sample()
        actions = actions.view(-1, ASIZE)
        action_log_probs = normal_distribution.log_prob(actions).sum(dim=-1)
        self.entropy = normal_distribution.entropy().mean()
        return actions, action_log_probs

    def reset_state(self):
        self.hidden = torch.zeros(12, RSIZE*2).to(self.device)

    def get_latent(self, mu, logsigma):
        """ Transform observations to latent space.

        :returns: (latent_obs, latent_next_obs)
            - latent: 4D torch tensor (BSIZE, SEQ_LEN, LSIZE)
        """
        latent = mu + logsigma.exp() * torch.randn_like(mu)
        return latent

    def VAE_loss(self, recon_x, x, mu, logsigma):
        """ VAE loss function """
        BCE = F.mse_loss(recon_x, x, reduction='none')
        BCE = BCE.mean(dim=0).sum()  # first mean up the batch dim
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * (1 + 2 * logsigma -
                      mu.pow(2) - (2 * logsigma).exp()).mean(0).sum()
        return BCE + KLD


class ActorCriticWorldModel(nn.Module):
    def __init__(self):
        super(ActorCriticWorldModel, self).__init__()

        num_actions = ASIZE
        # Setup the log std output for continuous action space
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))
        self.critic_linear = nn.Sequential(
            nn.Linear(LSIZE + RSIZE, LSIZE + RSIZE),
            nn.ReLU(),
            nn.Linear(LSIZE + RSIZE, 1),
        )
        self.actor_linear = nn.Sequential(
            nn.Linear(LSIZE + RSIZE, LSIZE + RSIZE),
            nn.ReLU(),
            nn.Linear(LSIZE + RSIZE, num_actions)
        )

        for param in self.parameters():
            if param.dim() > 1:
                nn.init.xavier_uniform_(param)

    def forward(self, x):

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, self.actor_logstd, value


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    test_model = World_model(device)
