import numpy as np 

class SarsaAgent: 
    def __init__(self,state_size,action_size,epsilon=0.1,alpha=0.5,gamma=0.95):
        self.state_size=state_size 
        self.action_size=action_size 
        self.epsilon=epsilon 
        self.alpha=alpha 
        self.gamma=gamma 

        self.Q=np.zeros((state_size,action_size)) 
    def get_action(self,state): 
        if np.random.random()<self.epsilon: 
            return np.random.randint(self.action_size) 
        else: 
            return np.argmax(self.Q[state]) 
    def learn(self,state,action,reward,next_state,next_action): 
        current_q=self.Q[state,action] 
        next_q=self.Q[next_state,next_action]

        new_q=current_q+self.alpha*(reward+self.gamma*next_q-current_q) 
        self.Q[state,action]=new_q 
    def train_sarsa(env,agent,episodes):
        for episode in range(episodes):
            state=env.reset() 
            action=agent.get_action(state) 
            done=False 
            while not done: 
                next_state,reward,done=env.step(action) 
                if not done: 
                    next_action=agent.get_action(next_state) 
                    agent.learn(state,action,reward,next_state,next_action) 
                    state=next_state
                    action=next_action 
                else: 
                    agent.learn(state,action,reward,next_state,0) 
                    