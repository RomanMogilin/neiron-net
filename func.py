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


def softmax_batch(t: list[list[float]]) -> list[list[float]]:
    result_list = []
    for t_item in t:
        result_list.append(softmax([t_item])[0])
    return result_list


# print(softmax_batch([[3, 5], [1, 2], [0, 9]]))


def cross_entropy(z: list[list[float]], y: list[list[float]]) -> list[list[float]]:
    entropy = 0
    for i in range(len(z[0])):
        entropy = entropy - y[0][i] * math.log(z[0][i])
    return entropy


def cross_entropy_batch(
    z: list[list[float]], y: list[list[float]]
) -> list[list[float]]:
    entropy_list = []
    for entropy_item_index in range(len(z)):
        entropy_list.append(
            cross_entropy([z[entropy_item_index]], [y[entropy_item_index]])
        )
    return entropy_list


# print(cross_entropy_batch([[5, 6], [3, 4]], [[8, 2], [1, 1]]))


def convert_y_in_stroke(y: float, num_of_classes: float) -> list[list[float]]:
    y_full = matrix_method.fill.zeros(num_of_classes, 1)
    y_full[0][y] = 1
    return y_full


def convert_y_to_matrix(y: list[float], num_of_classes: float) -> list[list[float]]:
    y_list = []
    for y_item_index in range(len(y)):
        y_list.append(convert_y_in_stroke(y[y_item_index], num_of_classes)[0])
    return y_list


# print(convert_y_to_matrix([5, 6, 3, 4], 8))


# test = softmax([[1, 6, 9, 0]])
# print(test[0].index(max(test[0])) + 1)
# print(cross_entropy(softmax([[1, 6, 9, 0]]), [[0, 0, 1, 0]]))
