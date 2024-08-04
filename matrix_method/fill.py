import random
import math


def zeros(column: int, row: int):
    return [[0 for _ in range(column)] for _ in range(row)]


def once(column: int, row: int):
    return [[1 for _ in range(column)] for _ in range(row)]


def rand(column: int, row: int):
    num_arr = []

    for i in range(row):
        num_arr.append([])
        for k in range(column):

            current_rand_num = math.sqrt((1 - random.random()) / random.random())
            znak = random.random()
            if znak >= 0.5:
                znak = -1
            else:
                znak = 1
            num_arr[i].append(znak * current_rand_num)
    # return [[random.random() * 2 - 1 for _ in range(column)] for _ in range(row)]
    return num_arr


def unic_number(column: int, row: int):
    return [[(i + 1) * (j + 1) + i for j in range(column)] for i in range(row)]


# print(unic_number(2, 3))
