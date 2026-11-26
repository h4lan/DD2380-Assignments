import sys
from math import log
from hmm1 import read_matrix

def safe_log(x):
    return -float("inf") if x <= 0 else log(x)

def partial_path_probability(A, B, pi, observations, time, state, delta=None, psi=None):
    N, T = len(pi), len(observations)
    if not delta:
        delta = [[0 for _ in range(N)] for _ in range(T)]
    if not psi:
        psi = [[-1 for _ in range(N)] for _ in range(T-1)]
    current_observation_proba = B[state][observations[time]]
    
    # Edge case: first time step
    if time == 0:
        delta[0][state] = safe_log(pi[state] * current_observation_proba)
        return delta[0][state]
    
    best_previous_delta, best_previous_state = -float("inf"), -1
    for j in range(N):
        if not delta[time-1][j]:
            delta[time-1][j] = partial_path_probability(A, B, pi, observations, time-1, j, delta, psi)
        d = delta[time-1][j] + safe_log(A[j][state]) 
        if d > best_previous_delta:
            best_previous_delta = d
            best_previous_state = j
    
    # Saving the previous state in the selected "most likely path"
    psi[time-1][state] = best_previous_state
    
    return best_previous_delta + safe_log(current_observation_proba)

def hmm2(input_text):
    # The lines have the same format as HMM1
    lines = input_text.strip().splitlines()
    A = read_matrix(lines[0])
    B = read_matrix(lines[1])
    pi = read_matrix(lines[2])[0]
    
    # The fourth line contains the number of observations T, followed by the T observations.
    tokens = lines[3].split()
    observations = [int(token) for token in tokens[1:]]
    
    N, T = len(pi), len(observations) # number of states and time steps
    
    # The psi matrix contains only T-1 rows, because the first state in the sequence doesn't have predecessors 
    delta = [[0 for _ in range(N)] for _ in range(T)]
    psi = [[-1 for _ in range(N)] for _ in range(T-1)]
    
    # Initialization: for each possible last state, pick the most likely path
    last_state_delta, best_last_state = -float("inf"), -1
    for i in range(N):
        p = partial_path_probability(A, B, pi, observations, T-1, i, delta, psi)
        if p > last_state_delta:
            best_last_state, last_state_delta = i, p 
    
    # Traversing the selected sequence backwards to retrieve the corresponding states
    sequence = [best_last_state]
    for i in range(T-2, -1, -1): #We retrieve each predecessor saved in psi
        sequence.insert(0, psi[i][sequence[0]])
    
    # Formatting output
    s = ""
    for state in sequence:
        s += f"{state} "
    s.strip()
    
    return s
        
    
if __name__ == "__main__":
    print(hmm2(sys.stdin.read()))
