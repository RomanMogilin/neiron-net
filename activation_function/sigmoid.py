import math


def func(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def derivative(x: float) -> float:
    return func(x) * (1 - func(x))
