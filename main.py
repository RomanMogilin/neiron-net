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

import neuron_net_functions

dataset = dataset_getter.get_dataset(60000)

# print(dataset)

# ==================== #

structure: neuron_net_functions.Structure = {
    "input_layer": 784,
    "hidden_layers": [128, 128, 128],
    "output_layer": 10,
}

learning_rate = 0.001

epoch_count = 50

batch_size = 50

activation_func = activation_function.sigmoid

# ==================== #

neuron_net: neuron_net_functions.neuronNet = neuron_net_functions.init_neuron_net(
    structure
)

loss_arr = []

for epoch in range(epoch_count):
    current_percent_of_complete_epochs = round((epoch / epoch_count) * 100, 3)
    print(f"process: {current_percent_of_complete_epochs:.2f}%")

    random.shuffle(dataset)

    accuracy = neuron_net_functions.calc_accuracy(
        dataset, neuron_net, structure, activation_func
    )

    print(f"Accuracy: {accuracy * 100:.2f}%")
    file = open("trained_neuron_net.txt", "w")
    file.write("neuron_net_final = " + str(neuron_net))
    file.close()
    print("file written")

    for batch_index in range(len(dataset) // batch_size):
        neuron_net_functions.percent_line(
            batch_index,
            (len(dataset) // batch_size),
            "batching",
        )
        batch = dataset[
            batch_index * batch_size : batch_index * batch_size + batch_size
        ]
        batch_matrix = func.convert_dataset_batch_in_matrix(batch)
        result_of_iteration = neuron_net_functions.one_iteration_of_training(
            neuron_net,
            batch_matrix["x"],
            func.convert_y_to_matrix(batch_matrix["y"], structure["output_layer"]),
            learning_rate,
            structure,
            activation_func,
        )

        neuron_net = result_of_iteration["neuron_net"]
        loss_arr.append(result_of_iteration["loss"])

        print("\033[F\033[K", end="")
    neuron_net_functions.percent_line(
        (len(dataset) // batch_size),
        (len(dataset) // batch_size),
        "batching",
    )
    print("\033[F\033[K", end="")
    print("\033[F\033[K", end="")
    print("\033[F\033[K", end="")
    print("\033[F\033[K", end="")
    print("\033[F\033[K", end="")

# write final version of tained neuron net #

file = open("trained_neuron_net.txt", "w")
file.write("neuron_net_final = " + str(neuron_net))
file.close()

# show plot #

import matplotlib.pyplot as plot

plot.plot(loss_arr)
plot.show()
