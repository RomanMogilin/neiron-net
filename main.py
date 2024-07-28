import activation_function
import activation_function.relu
import activation_function.sigmoid
import matrix_method
import matrix_method.fill
from typing import TypedDict, List, Dict
import func

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


def get_neuron_net_answer(
    x: list[list[float]], neuron_net: neuronNet
) -> list[list[float]]:
    z = x
    print("neuron_net: ", neuron_net)
    for layer in neuron_net.keys():
        current_layer: LayerParams = neuron_net[layer]
        if layer != layer_name(str(len(structure["hidden_layers"]) - 1), "output"):
            z = matrix_method.operation.element_operation(
                matrix_method.operation.plus(
                    matrix_method.operation.multiply(z, current_layer["w"]),
                    current_layer["b"],
                ),
                activation_function.relu.func,
            )
        else:
            # print(z)
            z = matrix_method.operation.plus(
                matrix_method.operation.multiply(z, current_layer["w"]),
                current_layer["b"],
            )
        print("z: ", z)
    return func.softmax(z)


structure: Structure = {"input_layer": 2, "hidden_layers": [3, 5], "output_layer": 4}

neuron_net: neuronNet = init_neuron_net(structure)

print(
    "soft_max z:",
    get_neuron_net_answer([[0, 9]], neuron_net),
    sum(get_neuron_net_answer([[0, 9]], neuron_net)[0]),
)
