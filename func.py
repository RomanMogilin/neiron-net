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
    for i in range(len(z[0])):
        entropy = entropy - y[0][i] * math.log(z[0][i])
    return entropy


def convert_y_in_stroke(y: float, num_of_classes: float) -> list[list[float]]:
    y_full = matrix_method.fill.zeros(num_of_classes, 1)
    y_full[0][y] = 1
    return y_full


test = softmax([[1, 6, 9, 0]])
print(test[0].index(max(test[0])) + 1)
# print(cross_entropy(softmax([[1, 6, 9, 0]]), [[0, 0, 1, 0]]))
