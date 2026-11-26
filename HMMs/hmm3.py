import sys
import hmm1
from hmm2 import safe_log

def matrix_times_vector(matrix, vector):
    result = []
    for i in range(len(matrix)):
        total = 0.0
        for j in range(len(vector)):
            total += matrix[i][j] * vector[j]
        result.append(total)
    return result

def scalar_multiplication(scalar, vector):
    result = []
    for value in vector:
        result.append(scalar * value)
    return result

# Forward pass: adapted from hmm1
def forward_pass(A, B, c, pi, observations):
    N, T = len(A), len(observations)
    alpha = [[0 for _ in range(N)] for _ in range(T)]
    
    # Step 1: Initialization, alpha_1(i) = pi[i] * B[i][O_1]
    first_observation = observations[0]
    emission_column = [row[first_observation] for row in B] # Take the column of B for the first observation.
    alpha[0] = hmm1.elementwise_product(pi, emission_column)
    c[0] = 0 if sum(alpha[0]) == 0 else 1/sum(alpha[0])
    alpha[0] = scalar_multiplication(c[0], alpha[0])

    # Step 2: Recursion
    t = 1 # Starting from time step 1 > 0
    for observation in observations[1:]:
        # For each current state, add the contributions from all possible previous states.
        predicted = hmm1.vector_times_matrix(alpha[t-1], A)  #alpha_t-1*A

        # Multiply each state value by its emission probability.
        emission_column = [row[observation] for row in B]
        alpha[t] = hmm1.elementwise_product(predicted, emission_column) #alpha_t= bitwise[(alpha_t-1*A)*B(:,Ot)^T] in vectors
            
        # Scaling alpha_t(i)
        c[t] = 0 if sum(alpha[t]) == 0 else 1/sum(alpha[t])
        alpha[t] = scalar_multiplication(c[t], alpha[t])
        t+=1
    
    return alpha

# Backward pass
def backward_pass(A, B, c, observations):
    N, T = len(A), len(observations)
    beta = [[0 for _ in range(N)] for _ in range(T)]
        
    # Step 1
    for i in range(N):
        beta[T-1][i] = c[T-1]
    
    # Step 2: Recursion
    t = T-2
    for observation in reversed(observations[1:]):
        emission_column = [row[observation] for row in B]
        beta[t] = scalar_multiplication(c[t], matrix_times_vector(A, hmm1.elementwise_product(beta[t+1], emission_column)))
        t-=1
        
    return beta

def reestimation(A, B, alpha, beta, observations):
    N, M, T = len(A), len(B[0]), len(observations)
    di_gamma = [[[0 for _ in range(N)] for _ in range(N)] 
             for _ in range(T)]
    gamma = [[0 for _ in range(N)] for _ in range(T)]
    
    # Step 1: using the alpha and beta matrices to compute gamma
    for t in range(T-1):
        for i in range(N):
            emission_column = [row[observations[t+1]] for row in B]
            result = hmm1.elementwise_product(scalar_multiplication(alpha[t][i], A[i]),
                                              hmm1.elementwise_product(beta[t+1], emission_column))
            for j in range(N):
                di_gamma[t][i][j] = result[j]
                gamma[t][i] += result[j]
                
    # Edge case: gamma for t=T-1
    for i in range(N):
        gamma[T-1][i] = alpha[T-1][i]
        
    # Step 2: reestimate the model
    updated_A = [[0 for _ in range(N)] for _ in range(N)]
    for i in range(N):
        denom = 0
        for t in range(T-1):
            denom += gamma[t][i]
        for j in range(N):
            numer = 0
            for t in range(T-1):
                numer += di_gamma[t][i][j]
            updated_A[i][j] = 0 if denom == 0 else numer/denom
            
    updated_B = [[-1 for _ in range(M)] for _ in range(N)]
    for i in range(N):
        denom = 0
        for t in range(T):
            denom += gamma[t][i]
        for j in range(M):
            numer = 0
            for t in range(T):
                if (observations[t] == j):
                    numer += gamma[t][i]
            updated_B[i][j] = 0 if denom == 0 else numer/denom
    
    return updated_A, updated_B

def hmm3(input_text):
    # The lines have the same format as HMM1
    lines = input_text.strip().splitlines()
    A = hmm1.read_matrix(lines[0])
    B = hmm1.read_matrix(lines[1])
    pi = hmm1.read_matrix(lines[2])[0]
    
    # The fourth line contains the number of observations T, followed by the T observations.
    tokens = lines[3].split()
    observations = [int(token) for token in tokens[1:]]
    
    N, M, T = len(A[0]), len(B[0]), len(observations)
    
    iters, maxIters = 0, 100
    old_log_prob = -float("inf")
    while True:
        c = [0 for i in range(T)]
        alpha = forward_pass(A, B, c, pi, observations)
        beta = backward_pass(A, B, c, observations)
        A, B = reestimation(A, B, alpha, beta, observations)

        log_prob = -sum(map(safe_log, c))
        iters += 1
        if (iters >= maxIters or log_prob <= old_log_prob):
            # Formatting output
            s = (f"{N} {N} " + ' '.join([str(A[i][j]) for i in range(N) for j in range(N)]) + "\n"
            + f"{N} {M} " + ' '.join([str(B[i][j]) for i in range(N) for j in range(M)]))
            return s
        
        # Else, iterate again
        old_log_prob = log_prob
        
if __name__ == "__main__":
    print(hmm3(sys.stdin.read()))
