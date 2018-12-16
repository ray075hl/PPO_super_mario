# PPO_super_mario
Play game super mario using Proximal Policy Optimization method.



## Setup

Tested environments is Ubuntu16.04 and Python3.6,  Pytorch>=0.4.0.

Other requirements package. 

```
pip install -r requirements.txt
```

Save video need install ffmpeg.

**Usage**

```bash
# Train a agent from scratch
python run.py train	
```

Download pre-trained model from [here](https://drive.google.com/file/d/1xUITZZ9u68XE3cufPSNhZAM0gyNDdqj2/view?usp=sharing).

```bash
# Play game with a trained model
python run.py play ./pre_trained_model/mario_10000-best.dat
```

Training processing takes about 5 hours when I use nvidia-V100(1GPU, 16 parallel game envs), rewards will reach about 200.0 and game length 275 steps. It look like below when model converge. 

![](./demo.gif)

## reference

* [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)
* [http://blog.varunajayasiri.com/ml/ppo.html](http://blog.varunajayasiri.com/ml/ppo.html) This great post help me a lot. It tell me how to warp a game  like deepmind done with atari game.
* [Game environment](https://github.com/Kautenja/gym-super-mario-bros)
* [PPO tutorial code](https://github.com/higgsfield/RL-Adventure-2) That is a very clean project and friendly for newbie rl algorithm learner. I borrow part code from it.

 