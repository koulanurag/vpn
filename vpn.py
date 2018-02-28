import argparse
import random
import torch
import torch.nn as nn
from torch.optim import RMSprop
import torch.multiprocessing as mp


class VPN(object):
    def __init__(self, network_fn, env_fn, d, k, n, max_memory, seed=0):
        """
        :param network_fn: callable function to get network instance
        :param env_fn: callable function to get environment instance
        :param d: Plan depth
        :param k: no. of prediction steps
        :param n:
        :param max_memory: Size of Replay Memory
        :param seed: seed for random generator
        """
        self.network_fn = network_fn
        self.env_fn = env_fn
        self.global_network = network_fn()
        self.target_network = network_fn()
        self.global_t = 0
        self.t = 0
        self.d = d
        self.k = k
        self.n = n
        self.max_memory = max_memory
        self.memory = []  # (state,option,reward,discount,next_state)
        self.global_optimizer = RMSprop(self.global_network.parameters())
        self.seed = seed
        self._stop_training = False

    def init_memory(self):
        self.memory = []  # (state,option,reward,discount,next_state)
        pass

    def train(self, num_processes):
        """Asynchronous n-step Q-learning with k-step prediction and d-step planning"""

        self.global_optimizer.zero_grad()
        self.global_t = 0
        self.t = 0
        self.init_memory()

        processes = []
        for rank in range(0, num_processes):
            p = mp.Process(target=self._train, args=(rank,))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()
        print('Training Over!')

    def update_global_grads(self, net):
        for param, shared_param in zip(net.parameters(), self.global_network.parameters()):
            if shared_param.grad is not None:
                return
            shared_param._grad = param.grad

    def _train(self, rank):
        def _execute_option(option):
            _option_reward = 0
            _done = False
            while option.next or not _done:
                _obs, _reward, _done, _ = env.step(option.next_action)
                _option_reward += reward
            return _option_reward, 0, _obs, _done

        torch.manual_seed(self.seed + rank)
        net = self.network_fn()
        net.load_state_dict(self.global_network.state_dict())

        t_start = self.t

        env = self.env_fn()
        obs = env.reset()
        done = False
        rewards, discounts = [], []
        while not done or (self.t - t_start) < self.n:
            if self.greedy_epsilon < random.random():
                option = None
            else:
                option = None
            reward, discount, next_state, done = _execute_option(option)
            self.t += 1
            self.global_t += 1
            rewards.append(reward)
            discounts.append(discount)

        R = 0 if done else max(1, 2)  # max q-value #thread-specific

        loss = 0
        for i in range(len(rewards) - 1, -1, -1):
            R = rewards[i] + discounts[i] * R
            reward_loss = None
            discount_loss = None
            value_loss = None
            loss += reward_loss + discount_loss + value_loss
            # note: only global network is updated and not thread-specific network
        t_dash = random.choice(range(0, self.max_memory - 1))
        for i in range(t_dash, t_dash + self.n):
            pass

        self.update_global_grads(net)
        if self.T % 10 == 0:
            self.target_network.load_state_dict(self.global_network.state_dict())

        if self._stop_training:
            return

    def test(self):
        convereged = False
        if convereged:
            self._stop_training = True

    @staticmethod
    def q_plan(net, state, option, depth, q_1=None, b=10):
        """
        :param state:
        :param option:
        :param depth:
        :param q_1:
        :param b:
        :return: Q-value from d-step planning
        """
        reward, discount, value, next_state = net(state, option)
        if depth == 1:
            return reward + discount * value
        best_options = get_best_options(net, state, b=10)
        option_q = []
        for option in best_options:
            option_q.append(VPN.q_plan(net, next_state, option, depth - 1))
        return reward + discount * (value + depth - 1 * max(option_q)) / depth


def get_best_options(net, state, b=10):
    return [1, 2, 3]


class VPNNetwork(nn.Module):
    def __init__(self, obs_space, n_actions):
        # encoder
        # value
        # outcome
        # transition
        pass

    def forward(self, input):
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Value Prediction Network Arguments')
    parser.add_argument('--train', type=bool, default=False)
    pass
