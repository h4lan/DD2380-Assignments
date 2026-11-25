def vector_times_matrix(vector, matrix):
    result = []
    for j in range(len(matrix[0])):
        total = 0.0
        for i in range(len(vector)):
            total += vector[i] * matrix[i][j]
        result.append(total)
    return result


#function that reads matrices form Kattis, It's applied to each line
def read_matrix(line):
    #split, the first two numbers are the number of rows and columns, next tokens are values 
    tokens = line.split()
    rows = int(tokens[0])
    cols = int(tokens[1])
    values = [float(token) for token in tokens[2:]]
    
    #add each row to the matrix, which goes from start (r*cols) to end (start+cols), where cols is read in input -> example: cols=3, [0*3,3],[1*3,6] etc
    matrix = []
    for r in range(rows): 
        start = r * cols
        end = start + cols
        row = values[start:end]
        matrix.append(row)
    return matrix

def hmm0(input_text):
    #extract lines and read all the matrices
    lines = input_text.strip().splitlines()
    transition = read_matrix(lines[0])
    emission = read_matrix(lines[1])
    initial = read_matrix(lines[2])[0]

    #calc as above p_t+1 and then O(t+1) vectors
    predicted_state = vector_times_matrix(initial, transition)
    predicted_observation = vector_times_matrix(predicted_state, emission)

    #from vectors to strings text
    return " ".join(["1", str(len(predicted_observation))] #rows(always 1) and col numbers 
                    + [str(value) for value in predicted_observation]) #values of predicted obs


import sys
print(hmm0(sys.stdin.read()))
