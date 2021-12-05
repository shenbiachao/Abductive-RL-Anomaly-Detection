from common import second_to_time_str
import numpy as np
from time import time
import torch
import random
import config


class ReplayBuffer(object):
    def __init__(self, obs_dim):
        self.max_buffer_size = config.max_buffer_size
        self.curr = 0
        self.gamma = config.gamma
        self.obs_dim = obs_dim
        self.action_dim = config.sample_size

        self.obs_buffer = np.zeros((config.max_buffer_size, self.obs_dim))
        self.action_buffer = np.zeros((config.max_buffer_size, self.action_dim))
        self.next_obs_buffer = np.zeros((config.max_buffer_size, self.obs_dim))
        self.reward_buffer = np.zeros((config.max_buffer_size,))
        self.done_buffer = np.zeros((config.max_buffer_size,))
        self.max_sample_size = 0

    def clear(self):
        self.obs_buffer = np.zeros((self.max_buffer_size, self.obs_dim))
        self.action_buffer = np.zeros((self.max_buffer_size, self.action_dim))
        self.next_obs_buffer = np.zeros((self.max_buffer_size, self.obs_dim))
        self.reward_buffer = np.zeros((self.max_buffer_size,))
        self.done_buffer = np.zeros((self.max_buffer_size,))
        self.max_sample_size = 0

    def add_traj(self, obs_list, action_list, next_obs_list, reward_list, done_list):
        for obs, action, next_obs, reward, done in zip(obs_list, action_list, next_obs_list, reward_list, done_list):
            self.add_tuple(obs, action, next_obs, reward, done)

    def add_tuple(self, obs, action, next_obs, reward, done):
        self.obs_buffer[self.curr] = obs
        self.action_buffer[self.curr] = action
        self.next_obs_buffer[self.curr] = next_obs
        self.reward_buffer[self.curr] = reward
        self.done_buffer[self.curr] = done

        # increase pointer
        self.curr = (self.curr + 1) % self.max_buffer_size
        self.max_sample_size = min(self.max_sample_size + 1, self.max_buffer_size)

    def sample_batch(self, batch_size, to_tensor=True, sequential=False):
        batch_size = min(self.max_sample_size, batch_size)
        if sequential:
            start_index = random.choice(range(self.max_sample_size), 1)
            index = []
            for i in range(batch_size):
                index.append((start_index + i) % self.max_sample_size)
        else:
            index = random.sample(range(self.max_sample_size), batch_size)
        obs_batch, action_batch, next_obs_batch, reward_batch, done_batch = \
            self.obs_buffer[index], self.action_buffer[index], self.next_obs_buffer[index], \
            self.reward_buffer[index], self.done_buffer[index]
        if to_tensor:
            obs_batch = torch.FloatTensor(obs_batch).to(config.device)
            action_batch = torch.FloatTensor(action_batch).to(config.device)
            next_obs_batch = torch.FloatTensor(next_obs_batch).to(config.device)
            reward_batch = torch.FloatTensor(reward_batch).to(config.device).unsqueeze(1)
            done_batch = torch.FloatTensor(done_batch).to(config.device).unsqueeze(1)
        return obs_batch, action_batch, next_obs_batch, reward_batch, done_batch


class DDPGTrainer():
    def __init__(self, agent, classifier, buffer, logger):
        self.agent = agent
        self.buffer = buffer
        self.logger = logger
        self.classifier = classifier

        self.batch_size = config.batch_size
        self.max_iteration = config.max_iteration
        self.steps_per_iteration = config.steps_per_iteration
        self.test_interval = config.test_interval
        self.num_test_trajectories = config.num_test_trajectories
        self.start_timestep = config.start_timestep
        self.log_interval = config.log_interval
        self.action_noise_scale = config.action_noise_scale

    def train(self):
        train_traj_rewards = [0]
        iteration_durations = []
        tot_steps = 0
        for ite in range(self.max_iteration):
            iteration_start_time = time()
            state = self.classifier.reset()
            traj_reward = 0
            for step in range(self.steps_per_iteration):
                action, _ = self.agent.select_action(state)
                action = action + np.random.normal(size=action.shape, scale=self.action_noise_scale)
                next_state, reward, done = self.classifier.step(action, tot_steps)
                traj_reward += reward
                self.buffer.add_tuple(state, action, next_state, reward, float(done))
                state = next_state

                tot_steps += 1
                if tot_steps < self.start_timestep:
                    continue

                data_batch = self.buffer.sample_batch(self.batch_size)
                loss_dict = self.agent.update(data_batch)

            iteration_end_time = time()
            iteration_duration = iteration_end_time - iteration_start_time
            iteration_durations.append(iteration_duration)

            train_traj_rewards.append(traj_reward)
            if ite % self.log_interval == 0:
                self.logger.log_var("reward/train", traj_reward, tot_steps)
                for loss_name in loss_dict:
                    self.logger.log_var(loss_name, loss_dict[loss_name], tot_steps)
            if ite % self.test_interval == 0:
                test_reward = self.test()
                self.logger.log_var("reward/test", test_reward, tot_steps)
                remaining_seconds = int((self.max_iteration - ite + 1) * np.mean(iteration_durations[-3:]))
                time_remaining_str = second_to_time_str(remaining_seconds)
                summary_str = "iteration {}/{}:\ttrain return {:.02f}\ttest return {:02f}\teta: {}". \
                    format(ite, self.max_iteration, train_traj_rewards[-1], test_reward, time_remaining_str)
                self.logger.log_str(summary_str)

    def test(self):
        rewards = []
        for episode in range(self.num_test_trajectories):
            traj_reward = 0
            state = self.classifier.reset()
            for step in range(self.steps_per_iteration):
                action, _ = self.agent.select_action(state, deterministic=True)
                next_state, reward, done, _ = self.classifier.step(action)
                traj_reward += reward
                state = next_state
                if done:
                    break
            rewards.append(traj_reward)

        return np.mean(rewards)
