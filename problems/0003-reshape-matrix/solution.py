import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape1: tuple[int, int]) -> list[list[int|float]]:
    #Write your code here and return a python list after reshaping by using numpy's tolist() method
    try:
        reshaped_matrix = np.reshape(a, new_shape1)
    except:
        return []
    return reshaped_matrix