import numpy as np 

class QlearningAgent: 
    def __init__(self, state_size, action_size, epsilon=0.1, alpha=0.1, gamma=0.95):
        self.state_size = state_size 
        self.action_size = action_size
        self.epsilon = epsilon 
        self.alpha = alpha 
        self.gamma = gamma 
        self.Q = np.zeros((state_size, action_size)) 

    def get_action(self, state):
        if np.random.random() < self.epsilon: 
            return np.random.randint(self.action_size) 
        else: 
            return np.argmax(self.Q[state]) 

    def learn(self, state, action, reward, next_state):
        current_q = self.Q[state, action]
        max_next_q = np.max(self.Q[next_state])
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.Q[state, action] = new_q 