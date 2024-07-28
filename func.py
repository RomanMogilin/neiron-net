import math
import matrix_method
import matrix_method.fill


def softmax(t: list[list[float]]) -> list[list[float]]:
    result = matrix_method.fill.zeros(len(t[0]), 1)
    sum_tj = 0
    for j in range(len(t[0])):
        sum_tj = sum_tj + math.exp(t[0][j])
    for i in range(len(t[0])):
        result[0][i] = math.exp(t[0][i]) / sum_tj
    return result


def cross_entropy(z: list[list[float]], y: list[list[float]]) -> list[list[float]]:
    entropy = 0
    for i in range(len(z)):
        entropy = entropy + y[i][0] * math.log(z[i][0])
    return -entropy


print(softmax([[1], [6], [9], [0]]))
print(cross_entropy(softmax([[1], [6], [9], [0]]), [[1], [0], [0], [0]]))
