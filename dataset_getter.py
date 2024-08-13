from tensorflow import keras
import tensorflow
import matplotlib.pyplot as plt


def get_dataset(count):
    mnist = keras.datasets.mnist

    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    train_dataset = tensorflow.data.Dataset.from_tensor_slices(
        (train_images, train_labels)
    )
    test_dataset = tensorflow.data.Dataset.from_tensor_slices(
        (test_images, test_labels)
    )

    dataset = []

    for image, label in train_dataset.take(count):
        array_to_list = []
        for i in image.numpy():
            for k in list(i):
                array_to_list.append(k / 255)

        dataset.append({"x": [array_to_list], "y": int(label.numpy())})

    return dataset


# data = get_dataset()
# # print(data)
# file = open("data.py", "w")
# file.write("data=" + str(data))
# file.close()
