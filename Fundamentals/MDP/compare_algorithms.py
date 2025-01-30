import numpy as np
import matplotlib.pyplot as plt
from mdp import value_iterations
from sarsa import SarsaAgent
from qlearning import QlearningAgent
from test_sarsa import GridWorldEnv as SarsaEnv
from test_sarsa import train_sarsa
from test_qlearning import train_qlearning

def create_grid_world_mdp(size=4):
    # MDP 환경 생성 (test_mdp.py에서 사용)
    S = [(i, j) for i in range(size) for j in range(size)]
    A = [0, 1, 2, 3]  # up, right, down, left
    T = {}
    R = {}
    
    for s in S:
        for a in A:
            for s_prime in S:
                T[(s_prime, s, a)] = 0.0
                i, j = s
                if a == 0:  # up
                    next_state = (max(0, i-1), j)
                elif a == 1:  # right
                    next_state = (i, min(size-1, j+1))
                elif a == 2:  # down
                    next_state = (min(size-1, i+1), j)
                else:  # left
                    next_state = (i, max(0, j-1))
                    
                if s_prime == next_state:
                    T[(s_prime, s, a)] = 1.0
                R[(s, a, s_prime)] = -1.0
    
    goal_state = (size-1, size-1)
    for a in A:
        for s_prime in S:
            R[(goal_state, a, s_prime)] = 10.0
    
    return S, A, T, R

def run_comparison(episodes=1000, size=4):
    # 1. Value Iteration (MDP)
    S, A, T, R = create_grid_world_mdp(size)
    V, pi_star_mdp = value_iterations(S, A, T, R, gamma=0.9)
    
    # 2. SARSA
    env_sarsa = SarsaEnv(size=size)
    sarsa_agent = SarsaAgent(
        state_size=size*size,
        action_size=4,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )
    sarsa_rewards = train_sarsa(env_sarsa, sarsa_agent, episodes)
    
    # 3. Q-Learning
    env_q = SarsaEnv(size=size)  # 같은 환경 사용
    q_agent = QlearningAgent(
        state_size=size*size,
        action_size=4,
        epsilon=0.1,
        alpha=0.1,
        gamma=0.9
    )
    q_rewards = train_qlearning(env_q, q_agent, episodes)
    
    # 결과 시각화
    plt.figure(figsize=(15, 5))
    
    # 1. 학습 곡선 비교
    plt.subplot(1, 2, 1)
    window = 50
    sarsa_avg = np.convolve(sarsa_rewards, np.ones(window)/window, mode='valid')
    q_avg = np.convolve(q_rewards, np.ones(window)/window, mode='valid')
    plt.plot(sarsa_avg, label='SARSA')
    plt.plot(q_avg, label='Q-Learning')
    plt.axhline(y=np.mean(list(V.values())), color='r', linestyle='--', label='Value Iteration')
    plt.xlabel('Episodes')
    plt.ylabel('Average Reward')
    plt.title('Learning Curves')
    plt.legend()
    
    # 2. 최종 정책 비교
    plt.subplot(1, 2, 2)
    action_symbols = ["↑", "→", "↓", "←"]
    
    policies = {
        'MDP': [[action_symbols[pi_star_mdp[(i,j)]] for j in range(size)] for i in range(size)],
        'SARSA': [[action_symbols[np.argmax(sarsa_agent.Q[i*size+j])] for j in range(size)] for i in range(size)],
        'Q-Learning': [[action_symbols[np.argmax(q_agent.Q[i*size+j])] for j in range(size)] for i in range(size)]
    }
    
    # 정책 표시
    cell_text = []
    for i in range(size):
        row = []
        for j in range(size):
            cell = f"M:{policies['MDP'][i][j]}\nS:{policies['SARSA'][i][j]}\nQ:{policies['Q-Learning'][i][j]}"
            row.append(cell)
        cell_text.append(row)
    
    plt.table(cellText=cell_text, loc='center', cellLoc='center')
    plt.title('Policy Comparison\n(M: MDP, S: SARSA, Q: Q-Learning)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

    # 수치적 비교 출력
    print("\n평균 보상 비교:")
    print(f"Value Iteration: {np.mean(list(V.values())):.2f}")
    print(f"SARSA 최종 100 에피소드 평균: {np.mean(sarsa_rewards[-100:]):.2f}")
    print(f"Q-Learning 최종 100 에피소드 평균: {np.mean(q_rewards[-100:]):.2f}")

if __name__ == "__main__":
    run_comparison() 