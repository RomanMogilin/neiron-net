import activation_function
import activation_function.relu
import activation_function.sigmoid
import matrix_method
import random
import matrix_method.fill
from typing import TypedDict, List, Dict
import func

import dataset_getter

import matrix_method.operation


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
            "b": matrix_method.fill.zeros(column, 1),
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


index_of_answer = 0


def get_neuron_net_answer(
    x: list[list[float]], neuron_net: neuronNet, structure: Structure, activation_func
):
    global index_of_answer
    file_of_debug = open("debug.txt", "w")
    z = x
    t_and_h: T_and_H = {}
    # print("neuron_net: ", neuron_net)
    for layer in neuron_net.keys():
        current_layer: LayerParams = neuron_net[layer]
        if layer != layer_name(str(len(structure["hidden_layers"]) - 1), "output"):
            # print(
            #     # "t:",
            #     # matrix_method.operation.multiply(z, current_layer["w"]),
            #     # "\nb:",
            #     # current_layer["b"],
            #     # len(z),
            #     # len(z[0]),
            #     # " >=< ",
            #     # len(current_layer["w"]),
            #     # len(current_layer["w"][0]),
            # )
            file_of_debug.write("z_" + str(index_of_answer) + ": " + str(z) + "\n")
            file_of_debug.write(
                "w_" + str(index_of_answer) + ": " + str(current_layer["w"]) + "\n"
            )
            t = matrix_method.operation.plus(
                matrix_method.operation.multiply(z, current_layer["w"]),
                current_layer["b"],
            )
            h = matrix_method.operation.element_operation(
                t,
                activation_func.func,
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
    # print("index_of_answer: ", index_of_answer)
    index_of_answer += 1
    file_of_debug.close()
    # print(z)
    return {"z": func.softmax(z), "t_and_h": t_and_h}


class Answer(TypedDict):
    z: List[List[float]]
    t_and_h: T_and_H


def one_iteration_of_training(
    neuron_net: neuronNet,
    x: list[list[float]],
    y: list[list[float]],
    learning_rate: float,
    structure: Structure,
    activation_func,
):
    copy_neuron_net: neuronNet = neuron_net

    # answer #

    answer_list: list[Answer] = []

    for answer_item in x:
        answer_list.append(
            get_neuron_net_answer(
                [answer_item], copy_neuron_net, structure, activation_func
            )
        )

    # === #

    # print(len(answer_list))

    # entropy, Z, Y #

    entropy = 0
    Z = []
    Y = []

    for entropy_index in range(len(answer_list)):
        entropy += func.cross_entropy(
            answer_list[entropy_index]["z"], [y[entropy_index]]
        )
        # print(answer_list[entropy_index]["z"], [y[entropy_index]])
        Z.append(answer_list[entropy_index]["z"][0])
        Y.append(y[entropy_index])

    loss = entropy
    # print(entropy)
    # print("Y:", Y)
    # print("Z:", Z)

    # === #

    # t_and_h #

    t_and_h: T_and_H = {}

    layers_names = list(copy_neuron_net.keys())

    # print(layers_names)

    for answer_layer in layers_names:
        t_and_h[answer_layer] = {"t": [], "h": []}

    for answer_item in answer_list:
        current_answer_item: T_and_H = answer_item["t_and_h"]
        # print(current_answer_item)
        for layer_name_answer in layers_names:
            current_answer_layer: T_and_H_elememt = current_answer_item[
                layer_name_answer
            ]
            t_and_h[layer_name_answer]["t"].append(current_answer_layer["t"][0])
            t_and_h[layer_name_answer]["h"].append(current_answer_layer["h"][0])

    # print(t_and_h)

    # === #

    # X #

    X = []

    for x_item in x:
        X.append(x_item)

    # print("X:", X, len(X), len(X[0]))

    # === #

    layers_reverse = list(copy_neuron_net.keys())
    layers_reverse.reverse()

    def dE_db_batch(b_matrix: list[list[float]]) -> list[list[float]]:
        b = matrix_method.fill.zeros(len(b_matrix[0]), 1)
        for b_item in b_matrix:
            b = matrix_method.operation.plus(b, [b_item])
        return b

    def change_w_and_b(
        dE_db: list[list[float]], dE_dw: list[list[float]], layer_name: str
    ) -> None:

        copy_neuron_net[layer_name]["w"] = matrix_method.operation.minus(
            copy_neuron_net[layer_name]["w"],
            matrix_method.operation.element_operation(
                dE_dw, lambda x: x * learning_rate
            ),
        )

        # print(">>>)))")
        # print(copy_neuron_net[layer_name]["b"])
        # copy_neuron_net[layer_name]["b"] = matrix_method.operation.minus(
        #     copy_neuron_net[layer_name]["b"],
        #     matrix_method.operation.element_operation(
        #         dE_db, lambda x: x * learning_rate
        #     ),
        # )

    if len(structure["hidden_layers"]) == 0:
        dE_dh = None
        dE_dt = matrix_method.operation.minus(Z, Y)
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(X), dE_dt
        )
        dE_db = dE_db_batch(dE_dt)

        change_w_and_b(dE_db, dE_dw, layer_name("input", "output"))
    else:
        first_layer_name = layer_name("input", "0")

        dE_dh = None
        dE_dt = matrix_method.operation.minus(Z, Y)
        # print(">>>>>>>>>>>>>>>>>")
        # print(Z, Y)
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(t_and_h[first_layer_name]["h"]),
            dE_dt,
        )
        # print(dE_dw)
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
                    t_and_h[layers_reverse[hidden_layer_name_index]]["t"],
                    activation_func.derivative,
                ),
            )
            dE_dw = matrix_method.operation.multiply(
                matrix_method.operation.transpose(t_and_h[next_layer_name]["h"]),
                dE_dt,
            )
            dE_db = dE_db_batch(dE_dt)

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
                t_and_h[layer_name("input", 0)]["t"],
                activation_func.derivative,
            ),
        )
        dE_dw = matrix_method.operation.multiply(
            matrix_method.operation.transpose(x),
            dE_dt,
        )
        dE_db = dE_db_batch(dE_dt)

        change_w_and_b(dE_db, dE_dw, layer_name("input", 0))

    return {"neuron_net": copy_neuron_net, "loss": loss}


import colorama

colorama.init()


def percent_line(current_count, count, message):
    num_of_tire = int(round((current_count / count) * 50))
    percent = round((current_count / count) * 100, 3)
    black_lines = int(50 - num_of_tire)
    print(
        colorama.Fore.GREEN
        + "━" * num_of_tire
        + colorama.Fore.BLACK
        + "━" * black_lines
        + " "
        + f"{percent:.2f}"
        + "%"
        + " "
        + message
    )


def calc_accuracy(
    dataset, neuron_net: neuronNet, structure: Structure, activation_func
):
    correct = 0
    for data_example_index in range(len(dataset)):
        percent_line(data_example_index, len(dataset), "calculating accuracy")
        x = dataset[data_example_index]["x"]
        y = dataset[data_example_index]["y"]
        answer: Answer = get_neuron_net_answer(
            x, neuron_net, structure, activation_func
        )
        y_predict = answer["z"][0].index(max(answer["z"][0]))
        # print(answer["z"])
        # print("y_predict:", y_predict, "y:", y)
        print("\033[F\033[K", end="")
        if y_predict == y:
            correct += 1
    percent_line(len(dataset), len(dataset), "calculating accuracy")
    acc = correct / len(dataset)

    return acc
