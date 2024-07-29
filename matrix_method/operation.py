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
    # print("===><><\n")
    # print(A)
    # print(B)
    # print("===><><\n")
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]

    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]

    return result


# print(multiply([[0, 9]], [[1, 2, 3], [4, 5, 6]]))


def element_multiply(A: list[list[float]], B: list[list[float]]) -> list[list[float]]:
    """
    matrix A element_multiply matrix B
    """
    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(A[i][j] * B[i][j])
        result.append(row)

    return result


def element_operation(A: list[list[float]], func) -> list[list[float]]:
    result = []

    for i in range(len(A)):
        row = []
        for j in range(len(A[0])):
            row.append(func(A[i][j]))
        result.append(row)

    return result


def transpose(A: list[list[float]]) -> list[list[float]]:
    """Транспонирует матрицу"""
    return [[row[i] for row in A] for i in range(len(A[0]))]
