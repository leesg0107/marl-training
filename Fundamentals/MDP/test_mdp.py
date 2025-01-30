import numpy as np
from mdp import value_iterations

def create_grid_world():
    # Define states (0,0) to (3,3) - total 16 states
    S = [(i, j) for i in range(4) for j in range(4)]
    
    # Define actions: up (0), right (1), down (2), left (3)
    A = [0, 1, 2, 3]
    
    # Initialize transition and reward dictionaries
    T = {}
    R = {}
    
    # Define transitions and rewards
    for s in S:
        for a in A:
            for s_prime in S:
                # Default probability is 0
                T[(s_prime, s, a)] = 0.0
                
                # Define next state based on action
                i, j = s
                if a == 0:  # up
                    next_state = (max(0, i-1), j)
                elif a == 1:  # right
                    next_state = (i, min(3, j+1))
                elif a == 2:  # down
                    next_state = (min(3, i+1), j)
                else:  # left
                    next_state = (i, max(0, j-1))
                
                # Set transition probability
                if s_prime == next_state:
                    T[(s_prime, s, a)] = 1.0
                
                # Define rewards
                R[(s, a, s_prime)] = -1.0  # small negative reward for each move
    
    # Set goal state reward
    goal_state = (3, 3)
    for a in A:
        for s_prime in S:
            R[(goal_state, a, s_prime)] = 10.0
    
    return S, A, T, R

def main():
    # Create grid world MDP
    S, A, T, R = create_grid_world()
    
    # Set discount factor
    gamma = 0.9
    
    # Run value iteration
    V, pi_star = value_iterations(S, A, T, R, gamma)
    
    # Print results
    print("\nOptimal Values:")
    for i in range(4):
        row = ""
        for j in range(4):
            row += f"{V[(i,j)]:6.2f} "
        print(row)
    
    print("\nOptimal Policy:")
    action_symbols = ["↑", "→", "↓", "←"]
    for i in range(4):
        row = ""
        for j in range(4):
            row += f"{action_symbols[pi_star[(i,j)]]:2s} "
        print(row)

if __name__ == "__main__":
    main() 