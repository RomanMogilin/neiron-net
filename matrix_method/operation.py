def plus(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    matrix A plus matrix B
    """

    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] + B[i][j])
        result.append(row)

    return result


def minus(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    matrix A minus matrix B
    """

    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] - B[i][j])
        result.append(row)

    return result


def multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    matrix A multiply matrix B
    """


def multiply(A, B):

    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result
