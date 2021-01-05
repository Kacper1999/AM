import numpy as np
from scipy.sparse import csr_matrix


def row_sparse_ge(csr_m: csr_matrix):
    (n, n) = csr_m.shape

    for k in range(n):
        mkk = csr_m[k, k]
        if mkk == 0:
            raise Exception("WRONG FORMAT 0 ON DIAGONAL")
        for ci in range(csr_m.indptr[k], csr_m.indptr[k + 1]):
            c = csr_m.indices[ci]
            csr_m[k, c] /= mkk
        for r in range(k + 1, n):
            mrk = csr_m[r, k]
            if mrk == 0:
                continue
            for ci in range(csr_m.indptr[r], csr_m.indptr[r + 1]):
                c = csr_m.indices[ci]
                csr_m[r, c] -= csr_m[k, c] * mrk

    print(csr_m.toarray())


def main():
    a = np.array([[1, 0, 0],
                  [0, 0, 3],
                  [4, 5, 0]], dtype=np.double)

    sa = csr_matrix(a)
    print(sa.data)
    print(sa.indices)
    print(sa.indptr)
    row_sparse_ge(sa)


if __name__ == '__main__':
    main()
