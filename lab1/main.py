import numpy as np
import time

def mlp_and_add(col, b):
    output = np.zeros(len(col))
    for i, c in enumerate(col):
        output[i] = c * b
    return output


def pji_matrix_mlp(a, b):
    output = np.zeros((a.shape[1], b.shape[1]))
    for i in range(a.shape[1]):
        for j in range(b.shape[1]):
            output[:, j] += mlp_and_add(a[:, i], b[i, j])
    return output


def main():
    # a = np.array([[1, 2, 3],
    #               [5, 6, 7],
    #               [8, 9, 0]])
    # b = np.array([[8],
    #               [9],
    #               [0]])
    #
    # print(pji_matrix_mlp(a, b))
    # print(a @ b)
    #
    # c = np.array([[1, 2, 3],
    #               [5, 6, 7],
    #               [8, 9, 0]])
    # d = np.array([[1, 2, 3],
    #               [5, 6, 7],
    #               [8, 9, 0]])
    #
    # print(pji_matrix_mlp(c, d))
    # print(c @ d)
    #

    for n in range(10, 110, 10):
        s = time.time_ns()
        a = np.random.rand(n, n)
        b = np.random.rand(n, n)
        pji_matrix_mlp(a, b)
        e = time.time_ns()
        print((e - s) / 1000000000)


if __name__ == '__main__':
    main()
