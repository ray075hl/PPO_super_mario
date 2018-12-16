import sys

from src.agent import Agent
from src.game_env import Game

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT


if __name__ == '__main__':
    obs_size = 84
    game_id = 'SuperMarioBrosNoFrameskip-1-1-v0'
    player = Agent((4, 84, 84), len(COMPLEX_MOVEMENT))

    if len(sys.argv) < 2:
        print('Usage: \tpython run.py train \n'
              '\t\tpython run.py play model_path')
        exit()

    if sys.argv[1] == 'train':
        env = Game(game_id, obs_size)
        player.train(env.game_id)

    if sys.argv[1] == 'play':
        if len(sys.argv) < 3:
            print('Usage: \tpython run.py train \n'
                  '\t\tpython run.py play model_path')
            exit()
        env = Game(game_id, obs_size, mode=sys.argv[1])
        player.play(sys.argv[2], env)
