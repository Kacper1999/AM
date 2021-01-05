import numpy as np


def pivoting(a, i):
    mj = i
    for j in range(i + 1, a.shape[0]):
        if a[j][i] > a[mj][i]:
            mj = j
    if a[mj][i] == 0:
        print("Macierz osobliwa")
        return
    if mj != i:
        tmp = a[i, i:].copy()
        a[i, i:] = a[mj, i:]
        a[mj, i:] = tmp


def gauss_elimination(a: np.array):
    output = a.astype(float)
    n = output.shape[0]
    for i in range(n):
        pivoting(output, i)
        for j in range(i + 1, n):
            output[j, i:n] -= output[i, i:n] * output[j, i] / output[i][i]
    return output


def main():
    a = np.array([[1, 2],
                  [3, 4]])
    print(gauss_elimination(a))

    b = np.array([[0.02, 0.01, 0, 0],
                  [1, 2, 1, 0],
                  [0, 1, 2, 1],
                  [0, 0, 100, 200]])
    print(gauss_elimination(b))

    c = np.array([[3, 2, 1],
                  [1, 2, 3],
                  [1, 4, 5]])
    print(gauss_elimination(c))


if __name__ == '__main__':
    main()
