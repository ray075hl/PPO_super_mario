import gym
import numpy as np
import cv2
import random

import multiprocessing

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_


DEBUG = False
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):  # len(rewards) = 10
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])

    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    # print(batch_size)
    for _ in range(batch_size // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[
                                                                                                       rand_ids, :]


def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                       returns, advantages):
            # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)
            dist, value = model(state)
            prob = F.softmax(dist, dim=-1)
            log_prob_ = F.log_softmax(dist, dim=-1)
            entropy = (prob*(-1.0*log_prob_)).mean()
            new_log_probs = F.log_softmax(dist, dim=1) #dist.log_prob(action)


            ratio = (new_log_probs[range(mini_batch_size), action.squeeze()] - old_log_probs[range(mini_batch_size), action.squeeze()]).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

            actor_loss =  -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            #print('critic_loss: {}, actor_loss: {}, entropy_loss: {}'.format(critic_loss, actor_loss, entropy))
            loss = 0.5*critic_loss + actor_loss - 0.001 * entropy

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

# class Game:
#
#     def __init__(self, seed: int):
#         # Breakout actions = ['noop', 'fire', 'right', 'left']
#         self.env = gym.make('BreakoutNoFrameskip-v4')
#
#         self.env.seed(seed)
#
#         self.obs_2_max = np.zeros((2, 1, 84, 84), np.uint8)
#
#         self.obs_4 = np.zeros((4, 84, 84))
#
#         self.rewards = []
#
#         self.lives = 0
#
#     def step(self, action):
#         reward = 0
#         done = None
#
#         # Each 4 frames take the same action
#         for i in range(4):
#             # obs is rgb picture
#             obs, r, done, info = self.env.step(action)
#
#             if i >= 2:
#                 self.obs_2_max[i % 2] = self._process_obs(obs)
#
#             reward += r
#
#             lives = self.env.unwrapped.ale.lives()
#
#             if lives < self.lives:
#                 done = True
#
#             self.lives = lives
#
#             # Game Over!
#             if done:
#                 break
#
#         self.rewards.append(reward)
#
#         # Game Over!
#         if done:
#
#             episode_info = {"reward": sum(self.rewards),
#                             "length": len(self.rewards)}
#
#             self.reset()
#         else:
#             episode_info = None
#
#             obs = self.obs_2_max.max(axis=0)
#             # cv2.imwrite('xxx.png', obs.transpose(1,2,0))
#
#             self.obs_4 = np.roll(self.obs_4, shift=-1, axis=0)
#             self.obs_4[-1:, ...] = obs
#
#         return self.obs_4, reward, done, episode_info
#
#
#     def reset(self):
#
#         self.env.reset()
#         obs, _, done, _ = self.env.step(1)  # 1 is fire button : signal of game begin
#         # self.obs_4 = np.zeros((4, 84, 84))
#         obs = self._process_obs(obs)
#         self.obs_4[0:, ...] = obs
#         self.obs_4[1:, ...] = obs
#         self.obs_4[2:, ...] = obs
#         self.obs_4[3:, ...] = obs
#
#         self.lives = self.env.unwrapped.ale.lives()
#
#         return self.obs_4
#
#     @staticmethod
#     def _process_obs(obs):
#
#         obs = cv2.cvtColor(obs, cv2.COLOR_BGR2GRAY)
#         obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
#         return obs[None, :, :]


class Game(object):



    def __init__(self, seed: int):



        self.env = gym.make('BreakoutNoFrameskip-v4')
        self.env.seed(seed)



        self.obs_2_max = np.zeros((2, 84, 84, 1), np.uint8)



        self.obs_4 = np.zeros((84, 84, 4))



        self.rewards = []



        self.lives = 0



    def step(self, action):

        reward = 0.
        done = None

        for i in range(4):

            obs, r, done, info = self.env.step(action)

            if i >= 2:
                self.obs_2_max[i % 2] = self._process_obs(obs)

            reward += r



            lives = self.env.unwrapped.ale.lives()
            # print(lives)

            if lives < self.lives:
                done = True
            self.lives = lives


            if done:
                break


        self.rewards.append(reward)

        if done:
            episode_info = {"reward": sum(self.rewards),
                            "length": len(self.rewards)}
            self.reset()
        else:
            episode_info = None

            obs = self.obs_2_max.max(axis=0)

            self.obs_4 = np.roll(self.obs_4, shift=-1, axis=-1)
            self.obs_4[..., -1:] = obs

        return self.obs_4, reward, done, episode_info


    def reset(self):


        obs = self.env.reset()
        obs, _, _, _ = self.env.step(1)

        obs = self._process_obs(obs)

        self.obs_4[..., 0:] = obs
        self.obs_4[..., 1:] = obs
        self.obs_4[..., 2:] = obs
        self.obs_4[..., 3:] = obs
        self.rewards = []

        self.lives = self.env.unwrapped.ale.lives()

        return self.obs_4




    @staticmethod
    def _process_obs(obs):



        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]  # Shape (84, 84, 1)



def worker_process(remote,
                   seed: int):

    game = Game(seed)

    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            remote.send(game.step(data))
        elif cmd == 'reset':
            remote.send(game.reset())
        elif cmd == 'close':
            remote.close()
        else:
            raise NotImplementedError


class Worker:

    #child: multiprocessing.Connection
    process: multiprocessing.Process

    def __init__(self, seed):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, seed))
        self.process.start()


class Model(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )


        conv_out_size = self._get_conv_out(input_shape)

        self.policy = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        fx = x.float() / 256
        conv_out = self.conv(fx).view(fx.size()[0],-1)
        return self.policy(conv_out), self.value(conv_out)

def play_game(net, game_: Game, visual=False):

    state = game_.reset()


    done = False
    total_reward = 0.0

    while not done:
        if visual == True:
            game_.env.render()
        state = torch.FloatTensor(state.transpose(2,0,1)).unsqueeze(0).to(device)
        dist, _ = net(state)
        dist = F.softmax(dist, dim=-1)
        action = np.argmax(dist.cpu().detach().numpy(), axis=-1)

        next_state, reward, done, _ = game_.step(action)

        state = next_state
        total_reward += reward

    return total_reward


if __name__ == '__main__':

    total_rewards_list = []
    action_space = 4
    lr = 0.0001
    gamma_ = 0.99
    lambda_ = 0.95

    updates = 10000

    ppo_epochs = 4
    n_workers = 8
    num_steps = 128
    n_mini_batch = 128

    batch_size = n_workers * num_steps

    mini_batch_size = batch_size // n_mini_batch
    assert (batch_size % n_mini_batch == 0)

    model = Model(input_shape=(4, 84, 84), n_actions=action_space).to(device)

    workers = [Worker(1111+i) for i in range(n_workers)]

    state = np.zeros((n_workers, 84, 84, 4), dtype=np.uint8)
    for worker in workers:
        worker.child.send(('reset', None))
    for i, worker in enumerate(workers):
        state[i] = worker.child.recv()
    state = state.transpose(0, 3, 1, 2)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_frames = 100000
    frame_idx = 0
    solved = False
    try:
        while frame_idx < max_frames and not solved:
            log_probs = []
            values = []
            states = []
            actions = []
            rewards = []
            masks = []


            for _ in range(num_steps):

                state = torch.FloatTensor(state).to(device)
                logits, value = model(state)

                prob = F.softmax(logits, dim=-1)
                action_list = []

                log_prob = F.log_softmax(logits, dim=-1)
                log_probs.append(log_prob)

                for i in range(len(prob)):
                    action = np.random.choice(action_space, p=prob.cpu().detach().numpy()[i])
                    action_list.append(action)
                # print(action_list)
                # print()

                for i, worker in enumerate(workers):
                    worker.child.send(("step", action_list[i]))
                next_state = []
                reward = []
                done = []
                for i, worker in enumerate(workers):
                    next_state_, reward_, done_, info = worker.child.recv()
                    next_state_ = next_state_.transpose(2, 0, 1)
                    next_state.append(next_state_[np.newaxis,...])
                    reward.append(reward_)
                    done.append(done_)

                next_state = np.concatenate(next_state, axis=0)
                reward = np.asarray(reward)
                done = np.asarray(done)

                # if DEBUG:
                #     print(next_state.shape)
                #     print(reward.shape)
                #     print(done.shape)
                #     print()

                values.append(value)
                rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))  # 2D list
                masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))  # 2D list

                states.append(state)
                actions.append(torch.from_numpy(np.asarray(action_list)).unsqueeze(1).to(device))
                state = next_state  # -----------------
            frame_idx += 1

            if frame_idx % 1 == 0:

                game_test = Game(random.randint(10000,20000))
                current = play_game(model, game_test)
                total_rewards_list.append(current)
                mean_last100 = sum(total_rewards_list[-100:])/len(total_rewards_list[-100:])
                print('frame_idx: {} \t mean_last100: {} \t current: {}'.format(frame_idx, mean_last100, current))
                if mean_last100 > 100:
                    solved = True

            next_state = torch.FloatTensor(next_state).to(device)
            _, next_value = model(next_state)

            returns = compute_gae(next_value, rewards, masks, values)
            returns = torch.cat(returns).detach()

            log_probs = torch.cat(log_probs).detach()
            values = torch.cat(values).detach()

            states = torch.cat(states)
            actions = torch.cat(actions)
            advantage = returns- values     # target_reward - predict_reward


            ppo_update(ppo_epochs, mini_batch_size, states, \
                       actions, log_probs, returns, advantage)
    finally:
        print('xixi')
        # End all process
        for w in workers:
            w.child.send(("close", None))



