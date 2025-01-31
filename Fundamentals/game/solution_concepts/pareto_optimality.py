import numpy as np

class ParetoOptimalityChecker:
    def __init__(self, payoff_matrices):
        self.payoff_matrices = payoff_matrices
        self.num_players = len(payoff_matrices)
        self.shape = payoff_matrices[0].shape
        
    def is_pareto_optimal(self, strategy_profile):
        current_payoffs = self.get_payoffs(strategy_profile)
        
        # 가능한 모든 전략 조합 검사
        for i in np.ndindex(self.shape):
            other_payoffs = self.get_payoffs(i)
            
            # 파레토 지배 여부 확인
            if self.dominates(other_payoffs, current_payoffs):
                return False
        return True
    
    def get_payoffs(self, strategy_profile):
        return [matrix[strategy_profile] for matrix in self.payoff_matrices]
    
    def dominates(self, payoffs1, payoffs2):
        # payoffs1이 payoffs2를 파레토 지배하는지 확인
        return (all(p1 >= p2 for p1, p2 in zip(payoffs1, payoffs2)) and 
                any(p1 > p2 for p1, p2 in zip(payoffs1, payoffs2)))
