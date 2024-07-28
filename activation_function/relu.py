def func(x: float) -> float:
    return max(0, x)


def derivative(x: float) -> float:
    if x >= 0:
        return 1
    elif x < 0:
        return 0
