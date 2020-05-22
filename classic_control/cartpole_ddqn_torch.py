#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from collections import namedtuple, deque
import random

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
NUM_EPISODES = 500

BATCH_SIZE = 32
CAPACITY = 1_000_000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def display_frames_as_gif(frames):
    plt.figure(figsize=(frames[0].shape[1] / 72, frames[0].shape[0] / 72), dpi=72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), lambda x: patch.set_data(frames[x]), frames=len(frames), interval=50)

    writer = animation.PillowWriter(fps=30)
    anim.save('movie_cartpole_DQN.gif', writer=writer)


def sample(memory, batch_size=BATCH_SIZE):
    return random.sample(memory, batch_size)


class Net(nn.Module):

    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hidden)
        self.fc2 = nn.Linear(n_hidden, n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_out)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        output = self.fc3(h2)
        return output


class Brain:

    def __init__(self, num_states, num_actions, lr=1e-4):
        self.num_actions = num_actions
        self.memory = deque(maxlen=CAPACITY)

        # 신경망 구성
        n_in, n_hidden, n_out = num_states, 32, num_actions
        self.main_q_network = Net(n_in, n_hidden, n_out)
        self.target_q_network = Net(n_in, n_hidden, n_out)
        print(self.main_q_network)

        self.optimizer = optim.Adam(self.main_q_network.parameters(), lr=lr)

    def replay(self):
        # --------------------------
        # 1. 저장된 transition 수 확인
        # --------------------------
        if len(self.memory) < BATCH_SIZE:
            return

        # --------------------------
        # 2. 미니배치 생성
        # --------------------------
        (self.batch,
         self.state_batch,
         self.action_batch,
         self.reward_batch,
         self.non_final_next_states) = self.make_minibatch()

        # --------------------------
        # 3. 정답신호로 사용할 Q(s_t, a_t)를 계산
        # --------------------------
        self.expected_state_action_values = self.get_expected_state_action_values()

        # --------------------------
        # 4. 결합 가중치 수정
        # --------------------------
        self.update_main_q_network()

    def decide_action(self, state, episode):
        # 현재 상태에 따라 행동을 결정한다.
        # ε-greedy 알고리즘에서 서서히 최적행동의 비중을 늘린다.
        epsilon = 0.5 * (1 / (episode + 1))

        if epsilon <= np.random.uniform(0, 1):
            self.main_q_network.eval()   # 신경망을 추론 모드로 전환
            with torch.no_grad():
                action = self.main_q_network(state).max(1)[1].view(1, 1)
            # 신경망 출력의 최댓값에 대한 인덱스 = max(1)[1]
            # .view(1, 1)은 [torch.LongTensor of size 1]을 size 1*1로 변환하는 역할
        else:
            action = torch.LongTensor([[random.randrange(self.num_actions)]])
        return action

    def make_minibatch(self):
        # 2.1 메모리 객체에서 미니배치를 추출
        transitions = sample(self.memory, BATCH_SIZE)

        # 2.2 각 변수를 미니배치에 맞는 형태로 변형
        # transitions는 각 단계별로 (state, action, state_next, reward) 형태로 BATCH_SIZE 개수만큼 저장됨
        # 다시 말해, (state, action, state_next, reward) * BATCH_SIZE 형태가 된다
        # 이를 미니배치로 만들기 위해
        # (state * BATCH_SIZE, action * BATCH_SIZE, state_next * BATCH_SIZE, reward * BATCH_SIZE) 형태로 변환한다
        batch = Transition(*zip(*transitions))

        # 2.3 각 변수의 요소를 미니배치에 맞게 변형하고, 신경망으로 다룰 수 있게 Variable로 만든다.
        # state를 예를 들면, [torch.FloatTensor of size 1*4] 형태의 요소가 BATCH_SIZE 개수만큼 있는 형태다.
        # 이를 torch.FloatTensor of size BATCH_SIZE * 4 형태로 변형한다.
        # 상태, 행동, 보상, non_final 상태로 된 미니배치를 나타내는 Variable을 생성
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        return batch, state_batch, action_batch, reward_batch, non_final_next_states

    def get_expected_state_action_values(self):
        # 3.1 신경망을 추론 모드로 전환
        self.main_q_network.eval()
        self.target_q_network.eval()

        # 3.2 신경망으로 Q(s_t, a_t)를 계산
        # self.model(state_batch)은 왼쪽, 오른쪽에 대한 Q값을 출력하여 [torch.FloatTensor of size BATCH_SIZE * 2] 형태다.
        # 여기서부터는 실행한 행동 a_t에 대한 Q값을 계산하므로 action_batch에서 취한 행동 a_t가 왼쪽이냐 오른쪽이냐에 대한
        # 인덱스를 구하고, 이에 대한 Q값을 gather 메소드로 모아온다.
        self.state_action_values = self.main_q_network(self.state_batch).gather(1, self.action_batch)

        # 3.3 max{Q(s_t+1, a)} 값을 계산한다. 이때 다음 상태가 존재하는지에 주의해야 한다.
        # cartpole이 done 상태가 아니고, next_state가 존재하는지 확인하는 인덱스 마스크를 만듦.
        non_final_mask = torch.BoolTensor(tuple(map(lambda s: s is not None, self.batch.next_state)))

        # 먼저 전체를 0으로 초기화
        next_state_values = torch.zeros(BATCH_SIZE)
        a_m = torch.zeros(BATCH_SIZE).type(torch.LongTensor)

        # 다음 상태에서 Q값이 최대가 되는 행동 a_m을 Main Q-Network로 계산
        # 마지막에 붙은 [1]로 행동에 해당하는 인덱스를 구함
        a_m[non_final_mask] = self.main_q_network(self.non_final_next_states).detach().max(1)[1]

        # 다음 상태가 있는 것만을 골라내고, size 32를 32 * 1로 변환
        a_m_non_final_next_states = a_m[non_final_mask].view(-1, 1)

        # 다음 상태가 있는 인덱스에 대해 행동 a_m의 Q값을 target Q-Network로 계산
        # detach() 메소드로 값을 꺼내옴
        # squeeze() 메소드로 size[minibatch*1]을 [minibatch]로 변환
        next_state_values[non_final_mask] = self.target_q_network(self.non_final_next_states).gather(1, a_m_non_final_next_states).detach().squeeze()

        # 3.4 정답신호로 사용할 Q(s_t, a_t) 값을 Q러닝 식으로 계산
        expected_state_action_values = self.reward_batch + GAMMA * next_state_values

        return expected_state_action_values

    def update_main_q_network(self):
        # 4.1 신경망을 학습 모드로 전환
        self.main_q_network.train()

        # 4.2 손실함수를 계산 (smooth_l1_loss는 Huber 함수)
        # expected_state_action_values는 size가 [minibatch]이므로 unsqueeze해서 [minibatch*1]로 만듦
        loss = F.smooth_l1_loss(self.state_action_values, self.expected_state_action_values.unsqueeze(1))

        # 4.3 결합 가중치를 수정
        self.optimizer.zero_grad()      # 경사를 초기화
        loss.backward()                 # 역전파 계산
        self.optimizer.step()           # 결합 가중치 

    def update_target_q_network(self):
        self.target_q_network.load_state_dict(self.main_q_network.state_dict())


class Agent:

    def __init__(self, num_states, num_actions):
        self.brain = Brain(num_states, num_actions)

    def update_q_function(self):
        self.brain.replay()

    def get_action(self, state, episode):
        action = self.brain.decide_action(state, episode)
        return action

    def memorize(self, state, action, next_state, reward):
        transition = Transition(state, action, next_state, reward)
        self.brain.memory.append(transition)

    def update_target_q_network(self):
        self.brain.update_target_q_network()


class Environment:

    def __init__(self):
        self.env = gym.make(ENV)
        num_states = self.env.observation_space.shape[0]
        num_actions = self.env.action_space.n
        self.agent = Agent(num_states, num_actions)

    def run(self):
        episode_10_list = np.zeros(10)  # 최근 10에피소드동안 버틴 단계 수를 저장함
                                        # (평균 단계 수를 출력할 때 사용)
        complete_episodes = 0
        episode_final = False
        frames = []

        for episode in range(NUM_EPISODES):
            observation = self.env.reset()

            state = observation
            state = torch.from_numpy(state).type(torch.FloatTensor)
            state = torch.unsqueeze(state, 0)   # size 4를 1*4로 변환

            for step in range(MAX_STEPS):
                if episode_final is True:
                    frames.append(self.env.render(mode='rgb_array'))

                self.env.render()

                action = self.agent.get_action(state, episode)

                # 행동 a_t를 실행해 다음 상태 s_{t+1}과 done 플래그 값을 결정
                # action에 .item()을 호출해 행동 내용을 구함
                observation_next, _, done, _ = self.env.step(action.item())

                # 보상을 부여하고 episode의 종료 판정 및 state_next를 설정
                if done:
                    state_next = None

                    # 최근 10 에피소드에서 버틴 단계 수를 리스트에 저장
                    episode_10_list = np.hstack((episode_10_list[1:], step+1))

                    if step < 195:
                        reward = torch.FloatTensor([-1.0])  # 도중에 봉이 쓰러졌다면 페널티로 보상 -1을 부여
                        complete_episodes = 0   # 연속 성공 에피소드 기록을 초기화
                    else:
                        reward = torch.FloatTensor([1.0])
                        complete_episodes = complete_episodes + 1

                else:
                    reward = torch.FloatTensor([0.0])   # 그 외의 경우는 보상 0을 부여
                    state_next = observation_next
                    state_next = torch.from_numpy(state_next).type(torch.FloatTensor)
                    state_next = torch.unsqueeze(state_next, 0) # size 4를 size 1*4로 변환

                # 메모리에 경험을 저장
                self.agent.memorize(state, action, state_next, reward)

                # Experience Replay로 Q함수를 수정
                self.agent.update_q_function()

                # 관측 결과를 업데이트
                state = state_next

                # 에피소드 종료 처리
                if done:
                    print('%d Episode: Finished after %d steps : 최근 10 에피소드의 평균 단계 수 = %.1lf' % (episode, step+1, episode_10_list.mean()))
                    # DDQN
                    if episode % 2 == 0:
                        self.agent.update_target_q_network()
                    break

            if episode_final is True:
                # 애니메이션 생성 및 저장
                display_frames_as_gif(frames)
                break

            # 10 에피소드 연속으로 195단계를 버티면 태스크 성공
            if complete_episodes >= 10:
                print('10 에피소드 연속 성공')
                episode_final = True


if __name__ == "__main__":
    cartpole_env = Environment()
    cartpole_env.run()
