import multiprocessing

import numpy as np
import cv2

import random

from nes_py.wrappers import JoypadSpace #as BinarySpaceToDiscreteSpaceEnv
import gym_super_mario_bros
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT

from src.monitor import Monitor

class Game:

    def __init__(self, game_id, obs_size, skip_frame=4, mode='train'):
        self.game_id = game_id
        env = gym_super_mario_bros.make(game_id)
        temp_obs = env.reset()
        height, width, _ = temp_obs.shape
        self.env = JoypadSpace(env, COMPLEX_MOVEMENT)

        self.obs_last2max = np.zeros((2, obs_size, obs_size, 1), np.uint8)

        self.obstack = np.zeros((obs_size, obs_size, 4))
        self.rewards = []
        self.lives = 2
        self.skip = skip_frame
        self.mode = mode
        if self.mode == 'play':
            self.monitor = Monitor(width=width, height=height)

    def step(self, action, monitor=False):
        print(self.lives)
        reward = 0.0
        done = False

        for i in range(self.skip):
            obs, r, done, info = self.env.step(action)

            if self.mode == 'play':
                print('Take Action: \t', COMPLEX_MOVEMENT[action])
                self.monitor.record(obs)

            if i >= 2:
                self.obs_last2max[i % 2] = self._process_obs(obs)

            # super mario's reward is cliped in [-15.0, 15.0]
            reward += r / 15.0
            lives = info['life']

            if lives < self.lives:
                print(lives, self.lives)
                done = True
            print(done)
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
        self.obstack[..., 0:] = obs
        self.obstack[..., 1:] = obs
        self.obstack[..., 2:] = obs
        self.obstack[..., 3:] = obs
        self.rewards = []

        self.lives = 2

        return self.obstack

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


# Unit test
if __name__ == '__main__':
    Game('SuperMarioBrosNoFrameskip-1-1-v0', 84)