import random


def zeros(column: int, row: int):
    return [[0 for _ in range(column)] for _ in range(row)]


def once(column: int, row: int):
    return [[1 for _ in range(column)] for _ in range(row)]


def rand(column: int, row: int):
    return [[random.random() for _ in range(column)] for _ in range(row)]
