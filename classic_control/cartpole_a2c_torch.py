#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
import copy
import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import animation

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

ENV = 'CartPole-v0'
GAMMA = 0.99
MAX_STEPS = 200
NUM_EPISODES = 1000

NUM_PROCESSES = 32
NUM_ADVANCED_STEP = 5   # 총 보상을 계산할 때 Advantage 학습을 할 단계 수
# A2C 손실함수 계산에 사용되는 상수
value_loss_coef = 0.5
entropy_coef = 1e-2
max_grad_norm = 0.5

BATCH_SIZE = 32
CAPACITY = 1_000_000

TD_ERROR_EPSILON = 1e-4     # 오차에 더해줄 바이어스

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class RolloutStorage:

    def __init__(self, num_steps, num_processes, obs_shape):
        self.observations = torch.zeros(num_steps+1, num_processes, 4)
        self.masks = torch.ones(num_steps+1, num_processes, 1)
        self.rewards = torch.zeros(num_steps, num_processes, 1)
        self.actions = torch.zeros(num_steps, num_processes, 1).long()

        # 할인 총 보상 저장
        self.returns = torch.zeros(num_steps+1, num_processes, 1)
        self.index = 0

    def insert(self, current_obs, action, reward, mask):
        self.observations[self.index+1].copy_(current_obs)
        self.masks[self.index+1].copy_(mask)
        self.rewards[self.index].copy_(reward)
        self.actions[self.index].copy_(action)

        self.index = (self.index + 1) % NUM_ADVANCED_STEP

    def after_update(self):
        # Advantage 학습 단계만큼 단계가 진행되면 가장 새로운 transition을 index0에 저장
        self.observations[0].copy_(self.observations[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        # Advantage 학습 범위 안의 각 단계에 대해 할인 총 보상을 계산
        # 주의: 5번째 단계부터 거슬러 올라오며 계산
        # 주의: 5번째 단계가 Advantage1, 4번째 단계가 Advantage2 ...
        self.returns[-1] = next_value
        for ad_step in reversed(range(self.rewards.size(0))):
            self.returns[ad_step] = self.returns[ad_step+1] * GAMMA * self.masks[ad_step+1] + self.rewards[ad_step]


class Net(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.actor = nn.Linear(n_hidden, n_out)     # 행동을 결정하는 부분이므로 출력 개수는
                                                    # 행동의 가짓수
        self.critic = nn.Linear(n_hidden, 1)        # 상태가치를 출력하는 부분이므로 출력 개수는 1개

    def forward(self, x):
        # 신경망 순전파 계산을 정의
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        critic_output = self.critic(h2)     # 상태가치 계산
        actor_output = self.actor(h2)       # 행동 계산

        return critic_output, actor_output

    def act(self, x):
        # 상태 x로부터 행동을 확률적으로 결정
        value, actor_output = self(x)
        # dim=1이므로 행동의 종류에 대해 softmax를 적용
        action_probs = F.softmax(actor_output, dim=1)
        action = action_probs.multinomial(num_samples=1)    # dim=1이므로 행동의 종류에 대해 확률을 계산

        return action

    def get_value(self, x):
        # 상태 x로부터 상태가치를 계산
        value, actor_output = self(x)
        return value

    def evaluate_actions(self, x, actions):
        # 상태 x로부터 상태가치, 실제 행동 actions의 로그 확률, 엔트로피를 계산
        value, actor_output = self(x)
        log_probs = F.log_softmax(actor_output, dim=1)      # dim=1이므로 행동의 종류에 대해 확률을 계산
        action_log_probs = log_probs.gather(1, actions)     # 실제 행동의 로그 확률(log_probs)를 구함

        probs = F.softmax(actor_output, dim=1)              # dim=1이므로 행동의 종류에 대해 계산
        entropy = -(log_probs * probs).sum(-1).mean()

        return value, action_log_probs, entropy


class Brain:

    def __init__(self, actor_critic: Net):
        self.actor_critic = actor_critic
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=1e-2)

    def update(self, rollouts):
        # Advantage 함수의 대상이 되는 5단계 모두를 사용해서 수정
        obs_shape = rollouts.observations.size()[2:]    # torch.Size([4, 84, 84])
        num_steps = NUM_ADVANCED_STEP
        num_processes = NUM_PROCESSES

        values, action_log_probs, entropy = self.actor_critic.evaluate_actions(rollouts.observations[:-1].view(-1, 4), rollouts.actions.view(-1, 1))

        # 주의: 각 변수의 크기
        # rollouts.observations[:-1].view(-1, 4)    -> torch.Size([80, 4])
        # rollouts.actions.view(-1, 1)              -> torch.Size([80, 1])
        # values                                    -> torch.Size([80, 1])
        # action_log_probs                          -> torch.Size([80, 1])
        # entropy                                   -> torch.Size([])

        values = values.view(num_steps, num_processes, 1)   # torch.Size([5, 16, 1])
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        # Advantage (행동가치-상태가치) 계산
        advantages = rollouts.returns[:-1] - values     # torch.Size([5, 16, 1])

        # Critic의 loss 계산
        value_loss = advantages.pow(2).mean()

        # Actor의 gain 계산, 나중에 -1을 곱하면 loss가 된다
        action_gain = (action_log_probs * advantages.detach()).mean()
        # detach 메소드를 호출해서 advantages를 상수로 취급

        # 오차함수의 총합
        total_loss = (value_loss * value_loss_coef - action_gain - entropy * entropy_coef)

        # 결합 가중치 수정
        self.actor_critic.train()
        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(self.actor_critic.parameters(), max_grad_norm)
        # 결합 가중치가 한번에 너무 크게 변화하지 않도록 경사를 0.5 이하로 제한 (클리핑)

        self.optimizer.step()


class Environment:

    def run(self):
        # 동시 실행할 환경 수만큼 env를 생성
        envs = [gym.make(ENV) for _ in range(NUM_PROCESSES)]

        # 모든 에이전트가 공유하는 Brain 객체를 생성
        n_in = envs[0].observation_space.shape[0]   # 상태 변수의 수 = 4
        n_out = envs[0].action_space.n              # 행동 가짓수 = 2
        n_hidden = 32
        actor_critic = Net(n_in, n_hidden, n_out)
        global_brain = Brain(actor_critic)

        # 각종 정보를 저장하는 변수
        obs_shape = n_in
        current_obs = torch.zeros(NUM_PROCESSES, obs_shape)     # torch.Size([16, 4])
        rollouts = RolloutStorage(NUM_ADVANCED_STEP, NUM_PROCESSES, obs_shape)

        episode_rewards = torch.zeros([NUM_PROCESSES, 1])   # 현재 에피소드의 보상
        final_rewards = torch.zeros([NUM_PROCESSES, 1])     # 마지막 에피소드의 보상
        obs_np = np.zeros([NUM_PROCESSES, obs_shape])
        reward_np = np.zeros([NUM_PROCESSES, 1])
        done_np = np.zeros([NUM_PROCESSES, 1])
        each_step = np.zeros(NUM_PROCESSES)                 # 각 환경의 단계 수를 기록
        episode = 0

        # 초기 상태로부터 시작
        obs = [envs[i].reset() for i in range(NUM_PROCESSES)]
        obs = np.array(obs)
        obs = torch.from_numpy(obs).float()     # torch.Size([16, 4])
        current_obs = obs

        # Advanced 학습에 사용되는 rollouts 객체에는 첫 번째 상태에 현재 상태를 저장
        rollouts.observations[0].copy_(current_obs)

        # 1 에피소드에 해당하는 반복문
        for j in range(NUM_EPISODES * NUM_PROCESSES):
            # Advanced 학습 대상이 되는 각 단계에 대해 계산
            for step in range(NUM_ADVANCED_STEP):
                # 행동을 선택
                with torch.no_grad():
                    action = actor_critic.act(rollouts.observations[step])

                # (16, 1) -> (16,)
                actions = action.squeeze(1).numpy()

                # 한 단계를 실행
                for i in range(NUM_PROCESSES):
                    obs_np[i], reward_np[i], done_np[i], _ = envs[i].step(actions[i])

                    # episode의 종료가치, state_next를 설정
                    if done_np[i]:
                        if i == 0:
                            print('%d Episode: Finished after %d steps' % (episode, each_step[i]+1))
                            episode += 1

                        # 보상 부여
                        if each_step[i] < 195:
                            reward_np[i] = -1.0
                        else:
                            reward_np[i] = 1.0

                        each_step[i] = 0
                        obs_np[i] = envs[i].reset()

                    else:
                        reward_np[i] = 0.0
                        each_step[i] += 1

                # 보상을 tensor로 변환하고, 에피소드의 총 보상에 더해줌
                reward = torch.from_numpy(reward_np).float()
                episode_rewards += reward

                # 각 실행 환경을 확인해서 done이 True이면 mask를 0으로, False면 mask를 1로
                masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done_np])

                # 마지막 에피소드의 총 보상을 업데이트
                final_rewards *= masks      # done이 False면 1을 곱하고, True면 0을 곱해 초기화
                final_rewards += (1 - masks) * episode_rewards

                # 에피소드의 총 보상을 업데이트
                episode_rewards *= masks

                # 현재 done이 True면 모두 0으로
                current_obs *= masks

                # current_obs를 업데이트
                obs = torch.from_numpy(obs_np).float()      # torch.Size([16, 4])
                current_obs = obs

                # 메모리 객체에 현 단계의 transition을 저장
                rollouts.insert(current_obs, action.data, reward, masks)

            # Advanced 학습 for문 끝

            # Advanced 학습 대상 중 마지막 단계의 상태로 예측하는 상태가치를 계산
            with torch.no_grad():
                next_value = actor_critic.get_value(rollouts.observations[-1]).detach()     # torch.Size([6, 16, 4])

            # 모든 단계의 할인 총 보상을 계산하고, rollouts의 returns 변수를 업데이트
            rollouts.compute_returns(next_value)

            # 신경망 및 rollout 업데이트
            global_brain.update(rollouts)
            rollouts.after_update()

            # 환경 개수를 넘어서는 횟수로 200 단계를 버텨내면 성공
            if final_rewards.sum().numpy() >= NUM_PROCESSES:
                print('연속 성공')
                break


if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()
