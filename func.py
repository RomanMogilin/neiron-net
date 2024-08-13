import math
import matrix_method
import matrix_method.fill


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
