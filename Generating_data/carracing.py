"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os.path import join, exists
import gym
import numpy as np
import matplotlib.pyplot as plt
import math

def sample_continuous_policy(action_space, seq_len, dt):
    """ Sample a continuous policy.

    Atm, action_space is supposed to be a box environment. The policy is
    sampled as a brownian motion a_{t+1} = a_t + sqrt(dt) N(0, 1).

    :args action_space: gym action space
    :args seq_len: number of actions returned
    :args dt: temporal discretization

    :returns: sequence of seq_len actions
    """
    actions = [action_space.sample()]
    for _ in range(seq_len):
        daction_dt = np.random.randn(*actions[-1].shape)
        actions.append(
            np.clip(actions[-1] + math.sqrt(dt) * daction_dt,
                    action_space.low, action_space.high))
    return actions

def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."
    from competitive_rl import make_envs

    # env = gym.make("CarRacing-v0")
    env = make_envs(
            env_id='cCarRacing-v0',
            seed=100,
            log_dir='data/dataset/',
            num_envs=1,
            asynchronous=False,
            resized_dim=96,
            action_repeat=1
        )

    seq_len = 1000

    for i in range(rollouts):
        env.reset()
        a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)
     

        s_rollout = []
        r_rollout = []
        d_rollout = []

        t = 0
        while True:
            action = a_rollout[t]
            t += 1
            # import pdb; pdb.set_trace()
            obs, r, done, _ = env.step(action.reshape(1,-1))
            # for ii in range(4):
            #     plt.figure()
            #     _obs = obs[0,ii,...]
            #     plt.imshow(_obs, cmap='gray', vmin=0, vmax=255)
            #     plt.savefig(f'./vis_env/{t}_{ii}.png')

            s_rollout += [obs[0]]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    import os 
    os.makedirs(args.dir, exist_ok=True)
    generate_data(args.rollouts, args.dir, args.policy)
