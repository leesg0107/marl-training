import numpy as np
from scipy.optimize import linprog

class NashEquilibriumSolver:
    def __init__(self, payoff_matrix):
        """
        payoff_matrix: 2D numpy array representing payoff matrix for player 1
        """
        self.payoff_matrix = payoff_matrix
        if len(payoff_matrix.shape) != 2:
            raise ValueError("Payoff matrix must be 2-dimensional")
        self.num_actions = payoff_matrix.shape[0]
    
    def solve(self):
        """
        Compute Nash equilibrium using linear programming
        """
        # 변수: 전략 확률 (num_actions) + value (1)
        c = np.zeros(self.num_actions + 1)
        c[-1] = -1  # Maximize value
        
        # 제약 조건 설정
        # 1) 각 전략에 대한 기대 보수가 value보다 작거나 같아야 함
        A_ub = np.zeros((self.num_actions, self.num_actions + 1))
        A_ub[:, :-1] = -self.payoff_matrix.T
        A_ub[:, -1] = 1
        b_ub = np.zeros(self.num_actions)
        
        # 2) 확률의 합이 1이어야 함
        A_eq = np.zeros((1, self.num_actions + 1))
        A_eq[0, :-1] = 1
        b_eq = np.array([1])
        
        # 3) 각 확률은 0 이상이어야 함
        bounds = [(0, None)] * self.num_actions + [(None, None)]
        
        result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds)
        
        if result.success:
            return result.x[:-1]  # 마지막 value 값을 제외한 전략 확률 반환
        else:
            raise ValueError("Failed to find Nash equilibrium")
