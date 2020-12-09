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
        # vae_file, rnn_file= \
        #     [join('data/lyc_world_model', m, 'best.tar')
        #      for m in ['vae', 'mdrnn']]

        # assert exists(vae_file) and exists(rnn_file),\
        #     "Either vae or mdrnn is untrained."

        # vae_state, rnn_state = [
        #     torch.load(fname)
        #     for fname in (vae_file, rnn_file)]

        # for m, s in (('VAE', vae_state), ('MDRNN', rnn_state)):
        #     print("Loading {} at epoch {} "
        #           "with test loss {}".format(
        #               m, s['epoch'], s['precision']))

        self.vae = VAE(4, LSIZE)
        # self.vae.load_state_dict(vae_state['state_dict'])

        self.mdrnn = MDRNNCell(LSIZE, ASIZE, RSIZE, 5)

        self.device = device
        self.actor_critic = ActorCriticWorldModel()
        self.hidden = torch.zeros(12, RSIZE*2).to(self.device)
        self.Loss = {'VAE_Loss': 0, 'MDN_Loss': 0}

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

        x = torch.cat((latent_mu, self.hidden[:, :RSIZE]), dim=1)
        logits, actor_logstd, value = self.actor_critic(x)
        actions = self.compute_action(logits, actor_logstd)
        # with torch.no_grad():
        mus, sigmas, logpi, _, _, next_hidden = self.mdrnn(
            actions, latent_mu, [self.hidden[:, :RSIZE], self.hidden[:, RSIZE:]])
        if loss:
            with torch.no_grad():
                next_obs_mu, next_obs_logsigma = self.vae(next_obs/255.0)[1:] 
                next_latent = self.get_latent(next_obs_mu, next_obs_logsigma)
            self.Loss['MDN_Loss'] = gmm_loss(
                next_latent, mus, sigmas, logpi)

        self.hidden = torch.cat(next_hidden, dim=1)
        return logits, actor_logstd, value

    def forward(self, obs, next_hidden=None, next_obs=None):

        logits, actor_logstd, value = self.get_action_and_transition(
            obs, next_hidden, next_obs)
        return logits, actor_logstd, value

    def compute_action(self, means, log_std):
        normal_distribution = torch.distributions.normal.Normal(
            means, torch.exp(log_std))
        actions = normal_distribution.sample()
        actions = actions.view(-1, ASIZE)
        return actions

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
        BCE = F.mse_loss(recon_x, x, size_average=False)
        # import pdb; pdb.set_trace()
        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        KLD = -0.5 * torch.sum(1 + 2 * logsigma -
                               mu.pow(2) - (2 * logsigma).exp())
        return BCE + KLD

    def MDNRNN_loss(self, latent_obs, action, reward, terminal,
                    latent_next_obs, include_reward: bool):
        """ Compute losses.

        The loss that is computed is:
        (GMMLoss(latent_next_obs, GMMPredicted) + MSE(reward, predicted_reward) +
            BCE(terminal, logit_terminal)) / (LSIZE + 2)
        The LSIZE + 2 factor is here to counteract the fact that the GMMLoss scales
        approximately linearily with LSIZE. All losses are averaged both on the
        batch and the sequence dimensions (the two first dimensions).

        :args latent_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor
        :args action: (BSIZE, SEQ_LEN, ASIZE) torch tensor
        :args reward: (BSIZE, SEQ_LEN) torch tensor
        :args latent_next_obs: (BSIZE, SEQ_LEN, LSIZE) torch tensor

        :returns: dictionary of losses, containing the gmm, the mse, the bce and
            the averaged loss.
        """
        latent_obs, action,\
            reward, terminal,\
            latent_next_obs = [arr.transpose(1, 0)
                               for arr in [latent_obs, action,
                                           reward, terminal,
                                           latent_next_obs]]
        mus, sigmas, logpi, rs, ds = mdrnn(action, latent_obs)
        gmm = gmm_loss(latent_next_obs, mus, sigmas, logpi)
        bce = F.binary_cross_entropy_with_logits(ds, terminal)
        if include_reward:
            mse = f.mse_loss(rs, reward)
            scale = LSIZE + 2
        else:
            mse = 0
            scale = LSIZE + 1
        loss = (gmm + bce + mse) / scale
        return dict(gmm=gmm, bce=bce, mse=mse, loss=loss)


class ActorCriticWorldModel(nn.Module):
    def __init__(self):
        super(ActorCriticWorldModel, self).__init__()

        num_actions = ASIZE
        # Setup the log std output for continuous action space
        self.actor_logstd = nn.Parameter(torch.zeros(1, num_actions))
        self.critic_linear = nn.Linear(LSIZE + RSIZE, 1)
        self.actor_linear = nn.Linear(LSIZE + RSIZE, num_actions)

    def forward(self, x):

        value = self.critic_linear(x)
        logits = self.actor_linear(x)

        return logits, self.actor_logstd, value
