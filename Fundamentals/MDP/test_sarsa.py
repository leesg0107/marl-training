import numpy as np
from sarsa import SarsaAgent

class GridWorldEnv:
    def __init__(self, size=4):
        self.size = size
        self.state = 0  # 시작 상태 (0,0)
        self.goal_state = size * size - 1  # 목표 상태 (3,3)
        
    def reset(self):
        self.state = 0
        return self.state
    
    def step(self, action):
        # action: 0(up), 1(right), 2(down), 3(left)
        row = self.state // self.size
        col = self.state % self.size
        
        # 다음 상태 계산
        if action == 0:  # up
            next_row = max(0, row - 1)
            next_col = col
        elif action == 1:  # right
            next_row = row
            next_col = min(self.size - 1, col + 1)
        elif action == 2:  # down
            next_row = min(self.size - 1, row + 1)
            next_col = col
        else:  # left
            next_row = row
            next_col = max(0, col - 1)
            
        self.state = next_row * self.size + next_col
        
        # 보상 및 종료 조건
        done = (self.state == self.goal_state)
        reward = 10 if done else -1
        
        return self.state, reward, done

def train_sarsa(env, agent, episodes):
    rewards_history = []
    
    for episode in range(episodes):
        state = env.reset()
        action = agent.get_action(state)
        total_reward = 0
        done = False
        
        while not done:
            next_state, reward, done = env.step(action)
            total_reward += reward
            
            if not done:
                next_action = agent.get_action(next_state)
                agent.learn(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            else:
                agent.learn(state, action, reward, next_state, 0)
        
        rewards_history.append(total_reward)
        
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f"Episode {episode + 1}, Average Reward: {avg_reward:.2f}")
    
    return rewards_history

def print_policy(agent, size=4):
    action_symbols = ["↑", "→", "↓", "←"]
    print("\nLearned Policy:")
    for i in range(size):
        row = ""
        for j in range(size):
            state = i * size + j
            action = np.argmax(agent.Q[state])
            row += f"{action_symbols[action]:2s} "
        print(row)

def print_q_values(agent, size=4):
    print("\nQ-Values:")
    for i in range(size):
        for j in range(size):
            state = i * size + j
            print(f"State ({i},{j}):")
            for action in range(4):
                print(f"  {['Up','Right','Down','Left'][action]}: {agent.Q[state,action]:.2f}")

def main():
    # 환경과 에이전트 초기화
    env = GridWorldEnv(size=4)
    agent = SarsaAgent(
        state_size=16,  # 4x4 그리드
        action_size=4,  # 상,우,하,좌
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )
    
    # 학습 실행
    episodes = 1000
    rewards_history = train_sarsa(env, agent, episodes)
    
    # 학습된 정책 출력
    print_policy(agent)
    
    # Q-값 출력
    print_q_values(agent)

if __name__ == "__main__":
    main() 