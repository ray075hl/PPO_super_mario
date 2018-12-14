import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

import numpy as np


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


def ppo_update(model, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.1):
    for _ in range(ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs,
                                                                         returns, advantages):
            pi, value = model(state)
            prob = F.softmax(pi, dim=-1)
            log_prob = F.log_softmax(pi, dim=-1)
            action_prob = prob.gather(1, action)

            action_prob_old = old_log_probs.exp().gather(1, action)

            ratio = action_prob / (action_prob_old + 1e-10)

            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-5)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, min=1. - clip_param, max=1. + clip_param) * advantage

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = (return_ - value).pow(2).mean()
            entropy_loss = (prob * log_prob).sum(1).mean()
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()


def adjust_learning_rate(optimizer, initial_lr, max_update_times, current_update_times):
    lr = initial_lr*(1 - 1.0*current_update_times/max_update_times)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
