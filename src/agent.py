import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
import os
from collections import deque


from src.game_env import *


is_cuda = torch.cuda.is_available()
device = 'gpu' if is_cuda else 'cpu'


class Net(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(Net, self).__init__()

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
        conv_out = self.conv(fx).view(fx.size()[0], -1)
        return self.policy(conv_out), self.value(conv_out)


class Agent:

    def __init__(self, obs_shape, action_space):

        self.policy = Net(obs_shape, action_space)


    def choose_action(self, logits, noise_action=True):
        # take an logits ---> action

        u = torch.rand(logits.size()).to(device)
        if noise_action:
            _, action = torch.max( logits.detach() - (-u.log()).log(), 1)
        else:
            logits = F.softmax(logits, dim=-1)
            action = np.argmax(logits.cpu().detach().numpy(), axis=-1)[0]

        return action


    def train(self, game_id,
                    obs_size=84,
                    mini_batch=4,
                    num_steps=128,
                    num_workers=16,
                    mini_epochs=3,
                    gamma_=0.99,
                    lambda_=0.95,
                    initial_lr=1e-4,
                    constant_lr=False,
                    max_update_times=10000,
                    early_stop=False,
                    save_model_steps=500):

        '''
        Agent training process!
        :param game_id:
        :param obs_size:
        :param mini_batch:
        :param num_steps:
        :param num_workers:
        :param mini_epochs:
        :param gamma_:
        :param lambda_:
        :param learning_rate:
        :param max_update_times:
        :param early_stop:
        :param save_model_steps:
        :return:
        '''
        batch_size = num_workers * num_steps

        mini_batch_size = batch_size // mini_batch
        assert (batch_size % mini_batch == 0)

        workers = [Worker(game_id, obs_size) for i in range(num_workers)]

        state = np.zeros((num_workers, obs_size, obs_size, 4), dtype=np.uint8)

        reward_queue = deque(maxlen=100)
        length_queue = deque(maxlen=100)

        for worker in workers:
            worker.child.send(('reset', None))

        for i, worker in enumerate(workers):
            state[i] = worker.child.recv()

        state = state.transpose(0, 3, 1, 2) # channel first for pytorch

        optimizer = optim.Adam(self.policy.parameters(), lr=initial_lr)

        current_update_times = 0
        while current_update_times < max_update_times and not early_stop:

            # ------------------------------------
            # interact with env and generate data
            rewards = []
            actions = []
            states = []

            logits, _ = self.policy(state)
            action = self.choose_action(logits)


            for worker in workers:
                worker.child.send('step', action)
            actions.append(action)


            # ------------------------------------


            # ------------------------------------
            #  update parameters



            # ------------------------------------




            # ------------------------------------
            # save model and print information

            # ------------------------------------


            pass






        # if not constant_lr:
        #     lr = adjust_learning_rate(optimizer, initial_lr, max_update_times, current_update_times)






    def play(self, model_path, game, visual=True, save_video=True):

        self.policy.load_state_dict(torch.load(model_path, map_location='cpu'))

        state = game.reset()
        done = False

        total_reward = 0.0

        while not done:

            if visual:
                game.env.render()

            logits, _ = self.policy(state.transpose(2, 0, 1)).unsqueeze(0).to(device)

            action = self.choose_action(logits, noise_action=False)

            next_state, reward, done, info = game.step(action, save_video)

            state = next_state
            total_reward += reward

            if done:
                game.env.reset()
                game.env.close()

        return total_reward, info['length']

    def save_model(self, update_step):
        filename = 'model_{}.dat'.format(update_step)
        torch.save(self.policy.state_dict(), os.path.join('checkpoint', filename))


# Unit test
if __name__ == '__main__':
    input_shape = (4, 84, 84)
    action_shape = 7
    model = Net(input_shape, action_shape)

