import sys

#same as hmm0
def read_matrix(line):
    # The first two numbers give the number of rows and columns.
    tokens = line.split()
    rows = int(tokens[0])
    cols = int(tokens[1])
    values = [float(token) for token in tokens[2:]]

    matrix = []
    for r in range(rows):
        start = r * cols
        end = start + cols
        matrix.append(values[start:end])

    return matrix

#same as hmm0
def vector_times_matrix(vector, matrix):
    result = []
    for j in range(len(matrix[0])):
        total = 0.0
        for i in range(len(vector)):
            total += vector[i] * matrix[i][j]
        result.append(total)
    return result

#NEW: create elemtwise product function: multiply the two values in each row.
def elementwise_product(vector1, vector2):
    result = []
    for i in range(len(vector1)):
        result.append(vector1[i] * vector2[i])
    return result

#Function for forward probability
def forward_probability(A, B, pi, observations):
    
    # Step 1: Initialization, alpha_1(i) = pi[i] * B[i][O_1]
    first_observation = observations[0]
    emission_column = [row[first_observation] for row in B] # Take the column of B for the first observation.
    alpha = elementwise_product(pi, emission_column)

    # Step 2: Recursion
    for observation in observations[1:]:
        # For each current state, add the contributions from all possible previous states.
        predicted = vector_times_matrix(alpha, A)  #alpha_t-1*A

        # Multiply each state value by its emission probability.
        emission_column = [row[observation] for row in B]
        alpha = elementwise_product(predicted, emission_column) #alpha_t= bitwise[(alpha_t-1*A)*B(:,Ot)^T] in vectors 

    # Step 3: Termination. Add the probabilities for all possible final states.
    return sum(alpha)


def hmm1(input_text):

    # The first three lines have the same format as in HMM0.
    lines = input_text.strip().splitlines()
    A = read_matrix(lines[0])
    B = read_matrix(lines[1])
    pi = read_matrix(lines[2])[0]

    # The fourth line contains the number of observations, followed by the T observations.
    tokens = lines[3].split()
    observations = [int(token) for token in tokens[1:]]

    return forward_probability(A, B, pi, observations)


if __name__ == "__main__":
    print(hmm1(sys.stdin.read()))