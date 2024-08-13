import math
import matrix_method
import matrix_method.fill


# def softmax(t: list[list[float]]) -> list[list[float]]:
#     result = matrix_method.fill.zeros(len(t[0]), 1)
#     sum_tj = 0
#     for j in range(len(t[0])):
#         sum_tj = sum_tj + math.exp(t[0][j])
#     for i in range(len(t[0])):
#         result[0][i] = math.exp(t[0][i]) / sum_tj
#     return result


def softmax(t: list[list[float]]) -> list[list[float]]:
    result = t
    max_t = max(t[0])
    result = matrix_method.operation.element_operation(t, lambda num: num - max_t)
    # print(result)
    sum_tj = 0
    for j in range(len(t[0])):
        sum_tj = sum_tj + math.exp(result[0][j])
    for i in range(len(t[0])):
        result[0][i] = math.exp(result[0][i]) / sum_tj
    return result


def softmax_batch(t: list[list[float]]) -> list[list[float]]:
    result_list = []
    for t_item in t:
        result_list.append(softmax([t_item])[0])
    return result_list


# print(softmax_batch([[3, 5], [1, 2], [0, 9]]))


def cross_entropy(z: list[list[float]], y: list[list[float]]) -> list[list[float]]:
    entropy = 0
    epsilon = 1e-15
    # print("z:", z)
    for i in range(len(z[0])):
        entropy = entropy - y[0][i] * math.log(z[0][i] + epsilon)
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


def convert_dataset_batch_in_matrix(
    batch,
) -> dict["x" : list[list[float]], "y" : list[int]]:
    x = []
    y = []
    for batch_item in batch:
        x.append(batch_item["x"][0])
        y.append(batch_item["y"])
    return {"x": x, "y": y}


# print(
#     convert_dataset_batch_in_matrix(
#         [
#             {"x": [[6.3, 2.5, 4.9, 1.5]], "y": 1},
#             {"x": [[7.7, 2.8, 6.7, 2.0]], "y": 2},
#             {"x": [[6.4, 3.2, 4.5, 1.5]], "y": 1},
#             {"x": [[5.7, 2.6, 3.5, 1.0]], "y": 1},
#             {"x": [[4.5, 2.3, 1.3, 0.3]], "y": 0},
#             {"x": [[5.5, 2.3, 4.0, 1.3]], "y": 1},
#             {"x": [[5.2, 2.7, 3.9, 1.4]], "y": 1},
#             {"x": [[6.9, 3.2, 5.7, 2.3]], "y": 2},
#             {"x": [[5.4, 3.9, 1.7, 0.4]], "y": 0},
#             {"x": [[6.1, 2.8, 4.0, 1.3]], "y": 1},
#             {"x": [[4.4, 2.9, 1.4, 0.2]], "y": 0},
#             {"x": [[7.6, 3.0, 6.6, 2.1]], "y": 2},
#             {"x": [[4.9, 3.6, 1.4, 0.1]], "y": 0},
#             {"x": [[6.0, 3.0, 4.8, 1.8]], "y": 2},
#             {"x": [[6.3, 3.3, 6.0, 2.5]], "y": 2},
#             {"x": [[5.6, 3.0, 4.1, 1.3]], "y": 1},
#             {"x": [[5.7, 3.0, 4.2, 1.2]], "y": 1},
#             {"x": [[7.2, 3.0, 5.8, 1.6]], "y": 2},
#             {"x": [[6.6, 3.0, 4.4, 1.4]], "y": 1},
#             {"x": [[6.1, 2.6, 5.6, 1.4]], "y": 2},
#             {"x": [[5.1, 3.7, 1.5, 0.4]], "y": 0},
#             {"x": [[5.7, 2.5, 5.0, 2.0]], "y": 2},
#             {"x": [[5.9, 3.2, 4.8, 1.8]], "y": 1},
#             {"x": [[5.9, 3.0, 4.2, 1.5]], "y": 1},
#             {"x": [[5.8, 2.7, 3.9, 1.2]], "y": 1},
#             {"x": [[4.7, 3.2, 1.6, 0.2]], "y": 0},
#             {"x": [[6.0, 3.4, 4.5, 1.6]], "y": 1},
#             {"x": [[6.3, 2.9, 5.6, 1.8]], "y": 2},
#             {"x": [[5.1, 3.5, 1.4, 0.2]], "y": 0},
#             {"x": [[5.1, 3.5, 1.4, 0.3]], "y": 0},
#             {"x": [[5.6, 3.0, 4.5, 1.5]], "y": 1},
#             {"x": [[5.8, 2.6, 4.0, 1.2]], "y": 1},
#             {"x": [[5.5, 2.6, 4.4, 1.2]], "y": 1},
#             {"x": [[5.7, 2.8, 4.5, 1.3]], "y": 1},
#             {"x": [[5.0, 2.3, 3.3, 1.0]], "y": 1},
#             {"x": [[5.5, 4.2, 1.4, 0.2]], "y": 0},
#             {"x": [[6.2, 2.8, 4.8, 1.8]], "y": 2},
#             {"x": [[7.4, 2.8, 6.1, 1.9]], "y": 2},
#             {"x": [[6.3, 2.8, 5.1, 1.5]], "y": 2},
#             {"x": [[4.6, 3.4, 1.4, 0.3]], "y": 0},
#             {"x": [[4.9, 3.0, 1.4, 0.2]], "y": 0},
#             {"x": [[6.8, 3.0, 5.5, 2.1]], "y": 2},
#             {"x": [[5.8, 2.7, 4.1, 1.0]], "y": 1},
#             {"x": [[5.9, 3.0, 5.1, 1.8]], "y": 2},
#             {"x": [[6.9, 3.1, 4.9, 1.5]], "y": 1},
#             {"x": [[6.3, 2.3, 4.4, 1.3]], "y": 1},
#             {"x": [[4.3, 3.0, 1.1, 0.1]], "y": 0},
#             {"x": [[7.3, 2.9, 6.3, 1.8]], "y": 2},
#             {"x": [[6.4, 2.8, 5.6, 2.2]], "y": 2},
#             {"x": [[7.7, 3.8, 6.7, 2.2]], "y": 2},
#         ]
#     )
# )

# print(convert_y_to_matrix([5, 6, 3, 4], 8))

# test = softmax([[1, 6, 9, 0]])
# print(test[0].index(max(test[0])) + 1)
# print(cross_entropy(softmax([[1, 6, 9, 0]]), [[0, 0, 1, 0]]))
