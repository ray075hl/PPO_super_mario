import multiprocessing

import numpy as np
import cv2

import random

from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


class Game:

    def __init__(self, game_id, obs_size, skip_frame=4):

        env = gym_super_mario_bros.make(game_id)
        self.env = BinarySpaceToDiscreteSpaceEnv(env, COMPLEX_MOVEMENT)

        self.obs_last2max = np.zeros((2, obs_size, obs_size, 1), np.uint8)

        self.obstack = np.zeros((obs_size, obs_size, 4))
        self.rewards = []
        self.lives = 3
        self.skip = skip_frame

    def step(self, action):
        reward = 0.0
        done = False

        for i in range(self.skip):
            obs, r, done, info = self.env.step(action)
            if i >= 2:
                self.obs_last2max[i % 2] = self._pocess_obs(obs)

            # super mario's reward is cliped in [-15.0, 15.0]
            reward += r/15.0
            lives = info['life']

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

            obs = self.obs_last2max.max(axis=0)

            self.obstack = np.roll(self.obstack, shift=-1, axis=-1)
            self.obstack[..., -1:] = obs

        return self.obstack, reward, done, episode_info

    def reset(self):
        obs = self.env.reset()

        obs = self._process_obs(obs)
        self.obs_4[..., 0:] = obs
        self.obs_4[..., 1:] = obs
        self.obs_4[..., 2:] = obs
        self.obs_4[..., 3:] = obs
        self.rewards = []

        self.lives = 3

        return self.obs_4

    @staticmethod
    def _process_obs(obs):

        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        return obs[:, :, None]


# Multiple environments for generate training data.
def worker_process(remote, game_id, obs_size):

    game = Game(game_id, obs_size)

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
    process: multiprocessing.Process

    def __init__(self, game_id, obs_size):
        self.child, parent = multiprocessing.Pipe()
        self.process = multiprocessing.Process(target=worker_process,
                                               args=(parent, game_id, obs_size))
        self.process.start()

