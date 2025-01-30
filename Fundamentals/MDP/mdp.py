import numpy as np 

def value_iterations(S,A,T,R,gamma,epsilon=1e-6):
    V={s:0 for s in S}
    while True:
        delta=0 
        V_prev=V.copy()

        for s in S:
            action_values=[] 
            for a in A:
                value=sum(T[(s_prime,s,a)]*(R[(s,a,s_prime)]+gamma*V_prev[s_prime])
                          for s_prime in S) 
                action_values.append(value) 
            V[s]=max(action_values) 
            delta=max(delta,abs(V[s]-V_prev[s])) 
        
        if delta < epsilon: 
            break 
            
    pi_star={} 
    for s in S:
        action_values=[]
        for a in A:
            value=sum(T[(s_prime,s,a)]*(R[(s,a,s_prime)]+gamma*V[s_prime]) for s_prime in S)
            action_values.append((value,a)) 
        pi_star[s]=max(action_values)[1]
    return V,pi_star 