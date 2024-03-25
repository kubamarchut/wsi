import numpy as np


def generate_input(minX: float, maxX: float, d: int) -> np.ndarray:
    return np.random.uniform(minX, maxX, d)
