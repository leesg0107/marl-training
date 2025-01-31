import numpy as np
from nash_equilibrium import NashEquilibriumSolver
from no_regret import NoRegretLearner
from pareto_optimality import ParetoOptimalityChecker
import matplotlib.pyplot as plt

def create_prisoner_dilemma():
    """
    죄수의 딜레마 게임 생성
    (배신, 배신): (-2, -2)
    (배신, 협력): (0, -3)
    (협력, 배신): (-3, 0)
    (협력, 협력): (-1, -1)
    """
    # 각 플레이어의 보수 행렬
    player1_payoff = np.array([
        [-2, 0],    # 배신 선택시
        [-3, -1]    # 협력 선택시
    ])
    player2_payoff = np.array([
        [-2, -3],   # 배신 선택시
        [0, -1]     # 협력 선택시
    ])
    return [player1_payoff, player2_payoff]

def test_nash_equilibrium(payoff_matrices):
    """내쉬 균형 테스트"""
    print("\n=== Nash Equilibrium Analysis ===")
    solver = NashEquilibriumSolver(payoff_matrices[0])  # 플레이어 1의 관점
    nash_strategy = solver.solve()
    print(f"Nash Equilibrium Strategy (Player 1): {nash_strategy}")
    
    expected_payoff = np.sum(nash_strategy * payoff_matrices[0].dot(nash_strategy))
    print(f"Expected Payoff at Nash Equilibrium: {expected_payoff:.2f}")

def test_no_regret(payoff_matrices, num_iterations=1000):
    """무후회 학습 테스트"""
    print("\n=== No-Regret Learning Analysis ===")
    learner1 = NoRegretLearner(num_actions=2)
    learner2 = NoRegretLearner(num_actions=2)
    
    action_history1 = []
    action_history2 = []
    
    for _ in range(num_iterations):
        # 각 플레이어의 행동 선택
        action1 = learner1.select_action()
        action2 = learner2.select_action()
        
        # 보상 계산
        reward1 = payoff_matrices[0][action1, action2]
        reward2 = payoff_matrices[1][action1, action2]
        
        # 다른 행동을 했을 때의 보상 계산
        other_rewards1 = np.array([payoff_matrices[0][a, action2] for a in range(2)])
        other_rewards2 = np.array([payoff_matrices[1][action1, a] for a in range(2)])
        
        # 학습 업데이트
        learner1.update(action1, reward1, other_rewards1)
        learner2.update(action2, reward2, other_rewards2)
        
        action_history1.append(learner1.get_action_probabilities()[0])
        action_history2.append(learner2.get_action_probabilities()[0])
    
    print("Final Strategy Probabilities:")
    print(f"Player 1 (Prob. of Cooperation): {action_history1[-1]:.2f}")
    print(f"Player 2 (Prob. of Cooperation): {action_history2[-1]:.2f}")
    
    return action_history1, action_history2

def test_pareto_optimality(payoff_matrices):
    """파레토 최적성 테스트"""
    print("\n=== Pareto Optimality Analysis ===")
    checker = ParetoOptimalityChecker(payoff_matrices)
    
    print("Checking all strategy profiles for Pareto optimality:")
    for i in range(2):
        for j in range(2):
            is_optimal = checker.is_pareto_optimal((i, j))
            action_names = ['Defect', 'Cooperate']
            print(f"Strategy Profile ({action_names[i]}, {action_names[j]}): {'Pareto Optimal' if is_optimal else 'Not Pareto Optimal'}")

def plot_no_regret_history(history1, history2):
    """무후회 학습 과정 시각화"""
    plt.figure(figsize=(10, 5))
    plt.plot(history1, label='Player 1 Cooperation Prob.')
    plt.plot(history2, label='Player 2 Cooperation Prob.')
    plt.xlabel('Iteration')
    plt.ylabel('Probability of Cooperation')
    plt.title('No-Regret Learning Process')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # 죄수의 딜레마 게임 생성
    payoff_matrices = create_prisoner_dilemma()
    
    # 각 솔루션 컨셉 테스트
    test_nash_equilibrium(payoff_matrices)
    history1, history2 = test_no_regret(payoff_matrices)
    test_pareto_optimality(payoff_matrices)
    
    # 학습 과정 시각화
    plot_no_regret_history(history1, history2)

if __name__ == "__main__":
    main() 