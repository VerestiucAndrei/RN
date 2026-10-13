import math


def load_system(path):
    A = []
    B = []
    file = open(path, "r")
    for line in file:
        words = line.split()
        row = []
        sign = 1
        for i in range(len(words) - 2):
            word = words[i]
            if word == "+":
                sign = 1
            elif word == "-":
                sign = -1
            else:
                number = word[:-1]
                if number == "":
                    number = "1"
                if number == "-":
                    number = "-1"
                row.append(sign * float(number))
        A.append(row)
        B.append(float(words[-1]))
    file.close()
    return A, B


def determinant(matrix):
    first = matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
    second = matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
    third = matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
    return first - second + third


def trace(matrix):
    return matrix[0][0] + matrix[1][1] + matrix[2][2]


def norm(vector):
    total = 0
    for x in vector:
        total = total + x * x
    return math.sqrt(total)


def transpose(matrix):
    result = []
    for j in range(3):
        row = []
        for i in range(3):
            row.append(matrix[i][j])
        result.append(row)
    return result


def multiply(matrix, vector):
    result = []
    for i in range(3):
        total = 0
        for j in range(3):
            total = total + matrix[i][j] * vector[j]
        result.append(total)
    return result


def solve_cramer(matrix, vector):
    det = determinant(matrix)
    solution = []
    for j in range(3):
        new_matrix = []
        for i in range(3):
            row = matrix[i].copy()
            row[j] = vector[i]
            new_matrix.append(row)
        solution.append(determinant(new_matrix) / det)
    return solution


def minor(matrix, i, j):
    result = []
    for r in range(3):
        if r != i:
            row = []
            for c in range(3):
                if c != j:
                    row.append(matrix[r][c])
            result.append(row)
    return result


def cofactor(matrix):
    result = []
    for i in range(3):
        row = []
        for j in range(3):
            m = minor(matrix, i, j)
            det = m[0][0] * m[1][1] - m[0][1] * m[1][0]
            row.append((-1) ** (i + j) * det)
        result.append(row)
    return result


def adjoint(matrix):
    return transpose(cofactor(matrix))


def solve(matrix, vector):
    det = determinant(matrix)
    adj = adjoint(matrix)
    inverse = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(adj[i][j] / det)
        inverse.append(row)
    return multiply(inverse, vector)


A, B = load_system("system.txt")
print("A =", A)
print("B =", B)
print("determinant =", determinant(A))
print("trace =", trace(A))
print("norm =", norm(B))
print("transpose =", transpose(A))
print("A * B =", multiply(A, B))
print("cramer =", solve_cramer(A, B))
print("inversion =", solve(A, B))
