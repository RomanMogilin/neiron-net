import activation_function
import activation_function.relu
import activation_function.sigmoid
import matrix_method
import matrix_method.fill
from typing import TypedDict, List, Dict
import func

import matrix_method.operation

from sklearn import datasets

iris = datasets.load_iris()

dataset = [(iris.data[i][None, ...], iris.target[i]) for i in range(len(iris.target))]

for learining_example in range(len(dataset)):
    dataset[learining_example] = {
        "x": [list(list(dataset[learining_example][0][0]))],
        "y": dataset[learining_example][1],
    }


class Structure(TypedDict):
    input_layer: int
    hidden_layers: List[int]
    output_layer: int


class LayerParams(TypedDict):
    w: List[List[int]]
    b: List[List[int]]


class neuronNet(TypedDict):
    Dict[str, LayerParams]


def layer_name(current_layer: str | int, next_layer: str | int) -> str:
    return "layer_%s_%s" % (current_layer, next_layer)


def init_neuron_net(structure: Structure) -> neuronNet:
    neuron_net = {}

    def fill_layer(key: str, row: int, column: int):
        neuron_net[key] = {
            "w": matrix_method.fill.rand(column, row),
            "b": matrix_method.fill.rand(column, 1),
        }

    if len(structure["hidden_layers"]) == 0:
        fill_layer(
            layer_name("input", "output"),
            structure["input_layer"],
            structure["output_layer"],
        )
    elif len(structure["hidden_layers"]) > 0:
        fill_layer(
            layer_name("input", "0"),
            structure["input_layer"],
            structure["hidden_layers"][0],
        )
        # print(
        #     "init: layer_name: %s, str_current: %s, structure_next: %s"
        #     % (
        #         layer_name("input", "0"),
        #         structure["input_layer"],
        #         structure["hidden_layers"][0],
        #     )
        # )
        if len(structure["hidden_layers"]) > 1:
            for layer in range(len(structure["hidden_layers"]) - 1):
                fill_layer(
                    layer_name(layer, layer + 1),
                    structure["hidden_layers"][layer],
                    structure["hidden_layers"][layer + 1],
                )
                # print(
                #     "init: layer_name: %s, str_current: %s, structure_next: %s"
                #     % (
                #         layer_name(layer, layer + 1),
                #         structure["hidden_layers"][layer],
                #         structure["hidden_layers"][layer + 1],
                #     )
                # )
        fill_layer(
            layer_name(str(len(structure["hidden_layers"]) - 1), "output"),
            structure["hidden_layers"][len(structure["hidden_layers"]) - 1],
            structure["output_layer"],
        )

    return neuron_net


class T_and_H_elememt(TypedDict):
    t: List[List[int]]
    h: List[List[int]]


class T_and_H(TypedDict):
    Dict[str, T_and_H_elememt]


def get_neuron_net_answer(x: list[list[float]], neuron_net: neuronNet):
    z = x
    t_and_h: T_and_H = {}
    # print("neuron_net: ", neuron_net)
    for layer in neuron_net.keys():
        current_layer: LayerParams = neuron_net[layer]
        if layer != layer_name(str(len(structure["hidden_layers"]) - 1), "output"):
            t = matrix_method.operation.plus(
                matrix_method.operation.multiply(z, current_layer["w"]),
                current_layer["b"],
            )
            h = matrix_method.operation.element_operation(
                t,
                activation_function.relu.func,
            )
            t_and_h[layer] = {"t": t, "h": h}
            z = h
        else:
            # print(z)
            t = matrix_method.operation.plus(
                matrix_method.operation.multiply(z, current_layer["w"]),
                current_layer["b"],
            )
            t_and_h[layer] = {"t": t, "h": [[]]}
            z = t
        # print("z: ", z)
    return {"z": func.softmax(z), "t_and_h": t_and_h}


# ==================== #


structure: Structure = {"input_layer": 4, "hidden_layers": [5], "output_layer": 3}

learning_rate = 0.001

epoch_count = 200

# ==================== #

neuron_net: neuronNet = init_neuron_net(structure)


class Answer(TypedDict):
    z: List[List[float]]
    t_and_h: T_and_H


loss_arr = []


def one_iteration_of_training(
    neuron_net: neuronNet,
    x: list[list[float]],
    y: list[list[float]],
    learning_rate: float,
):
    copy_neuron_net: neuronNet = neuron_net
    answer: Answer = get_neuron_net_answer(x, copy_neuron_net)

    entropy = func.cross_entropy(answer["z"], y)
    loss_arr.append(entropy)

    layers_reverse = list(copy_neuron_net.keys())
    layers_reverse.reverse()
    # print(layers_reverse)

    # print(neuron_net)

    def change_w_and_b(
        dE_db: list[list[float]], dE_dw: list[list[float]], layer_name: str
    ) -> None:

        # print(
        #     ">>>",
        #     copy_neuron_net[layer_name]["w"],
        #     "<><>",
        #     matrix_method.operation.element_operation(
        #         dE_dw, lambda x: x * learning_rate
        #     ),
        # )
        copy_neuron_net[layer_name]["w"] = matrix_method.operation.minus(
            copy_neuron_net[layer_name]["w"],
            matrix_method.operation.element_operation(
                dE_dw, lambda x: x * learning_rate
            ),
        )

        copy_neuron_net[layer_name]["b"] = matrix_method.operation.minus(
            copy_neuron_net[layer_name]["b"],
            matrix_method.operation.element_operation(
                dE_db, lambda x: x * learning_rate
            ),
        )

    if len(structure["hidden_layers"]) == 0:
        dE_dh = None
        dE_dt = matrix_method.operation.minus(answer["z"], y)
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(x), dE_dt
        )
        dE_db = dE_dt

        change_w_and_b(dE_db, dE_dw, layer_name("input", "output"))
    else:
        # print("answer['t_and_h']: ", answer["t_and_h"])
        # print(layers_reverse[1])
        dE_dh = None
        dE_dt = matrix_method.operation.minus(answer["z"], y)
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(
                answer["t_and_h"][layers_reverse[1]]["h"]
            ),
            dE_dt,
        )
        dE_db = dE_dt

        change_w_and_b(
            dE_db, dE_dw, layer_name(len(structure["hidden_layers"]) - 1, "output")
        )

        previous_layer_name = layers_reverse[0]
        next_layer_name = (
            layers_reverse[2] if len(layers_reverse) > 2 else layers_reverse[1]
        )
        #
        for hidden_layer_name_index in range(1, len(layers_reverse) - 1):
            next_layer_name = layers_reverse[hidden_layer_name_index + 1]
            # print("hidden_layer_name_index:", hidden_layer_name_index)
            dE_dh = matrix_method.operation.multiply(
                dE_dt,
                matrix_method.operation.transpose(
                    copy_neuron_net[previous_layer_name]["w"]
                ),
            )
            dE_dt = matrix_method.operation.element_multiply(
                dE_dh,
                matrix_method.operation.element_operation(
                    answer["t_and_h"][layers_reverse[hidden_layer_name_index]]["t"],
                    activation_function.relu.derivative,
                ),
            )
            dE_dw = matrix_method.operation.multiply(
                matrix_method.operation.transpose(
                    answer["t_and_h"][next_layer_name]["h"]
                ),
                dE_dt,
            )
            dE_db = dE_dt

            change_w_and_b(dE_db, dE_dw, layers_reverse[hidden_layer_name_index])
            # print(
            #     previous_layer_name,
            #     layers_reverse[hidden_layer_name_index],
            #     next_layer_name,
            # )
            previous_layer_name = layers_reverse[hidden_layer_name_index]

        #
        dE_dh = matrix_method.operation.multiply(
            dE_dt,
            matrix_method.operation.transpose(
                copy_neuron_net[previous_layer_name]["w"]
            ),
        )
        dE_dt = matrix_method.operation.element_multiply(
            dE_dh,
            matrix_method.operation.element_operation(
                answer["t_and_h"][layer_name("input", 0)]["t"],
                activation_function.relu.derivative,
            ),
        )
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(x),
            dE_dt,
        )
        dE_db = dE_dt

        change_w_and_b(dE_db, dE_dw, layer_name("input", 0))

    return copy_neuron_net


for epoch in range(epoch_count):
    for data_example_index in range(len(dataset)):
        x = dataset[data_example_index]["x"]
        y = dataset[data_example_index]["y"]
        neuron_net = one_iteration_of_training(
            neuron_net,
            x,
            func.convert_y_in_stroke(y, structure["output_layer"]),
            learning_rate,
        )
        # print(func.convert_y_in_stroke(y, structure["output_layer"]), y)

print(dataset)


def calc_accuracy():
    correct = 0
    for data_example_index in range(len(dataset)):
        x = dataset[data_example_index]["x"]
        y = dataset[data_example_index]["y"]
        answer: Answer = get_neuron_net_answer(x, neuron_net)
        y_predict = answer["z"][0].index(max(answer["z"][0]))
        # print(answer["z"])
        # print("y_predict:", y_predict, "y:", y)
        if y_predict == y:
            correct += 1

    acc = correct / len(dataset)

    return acc


accuracy = calc_accuracy()
print(neuron_net)
print("Accuracy:", accuracy * 100, "%")


# import matplotlib.pyplot as plot

# plot.plot(loss_arr)
# print(loss_arr.index(max(loss_arr)))
# print(loss_arr.index(min(loss_arr)))
# plot.show()


# print("===\nepoch:\n===\n")
# keys = list(structure.keys())
# keys.reverse()
# print(keys)
# print(
#     one_iteration_of_training(
#         neuron_net,
#         [[1, 9]],
#         func.convert_y_in_stroke(3, structure["output_layer"]),
#         learning_rate,
#     )
# )
