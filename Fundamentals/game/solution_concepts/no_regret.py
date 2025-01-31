import numpy as np

class NoRegretLearner:
    def __init__(self, num_actions, learning_rate=0.1):
        self.num_actions = num_actions
        self.lr = learning_rate
        self.weights = np.ones(num_actions)
        self.cumulative_regret = np.zeros(num_actions)
        
    def get_action_probabilities(self):
        return self.weights / np.sum(self.weights)
    
    def select_action(self):
        probs = self.get_action_probabilities()
        return np.random.choice(self.num_actions, p=probs)
    
    def update(self, action, reward, other_rewards):
        regret = np.maximum(0, other_rewards - reward)
        self.cumulative_regret += regret
        self.weights = np.exp(self.lr * self.cumulative_regret)
