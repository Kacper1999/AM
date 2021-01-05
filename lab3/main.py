import numpy as np


class ElementFormat:
    """I don't know how to specify row from the parameters from lecture
    so I changed elt_var so that it is the list of tuples (lowest_row_num, lowest_col_num, size)
    and I assume every block is the square matrix"""

    def __init__(self, elt_var, val):
        self.elt_var = elt_var
        self.val = val

    def to_csr(self):
        tmp = []
        i = 0
        for lrn, lcn, s in self.elt_var:
            for c in range(s):
                for r in range(s):
                    v = self.val[i]
                    tmp.append((lrn + r, lcn + c, v))
                    i += 1
        tmp.sort()

        pr, pc = -1, -1
        val = []
        icl = []
        row_ptr = []
        i = 0
        for (r, c, v) in tmp:
            if v == 0:
                continue
            if (pr, pc) == (r, c):
                val[-1] += v
                continue
            elif pr != r:
                row_ptr.append(i)
                for _ in range(r - pr - 1):
                    row_ptr.append(i)
            val.append(v)
            icl.append(c)
            i += 1
            pr, pc = r, c
        row_ptr.append(i)
        return CSRFormat(val, icl, row_ptr)


class CSRFormat:
    def __init__(self, val, icl, row_ptr):
        self.val = val
        self.icl = icl
        self.row_ptr = row_ptr

    def __str__(self):
        output = "val " + str(self.val) + "\n"
        output += "icl " + str(self.icl) + "\n"
        output += "row " + str(self.row_ptr)
        return output

    def mlp_by_vec(self, vec):
        output = [0] * (self.row_ptr[-1] - 1)
        for i, (v, c) in enumerate(zip(self.val, self.icl)):
            for r, j in enumerate(self.row_ptr):
                if j > i:
                    output[r - 1] += v * vec[c]
                    break
        return output


def main():
    a = np.array([[1, 0, 0],
                  [0, 2, 3],
                  [0, 0, 5]])
    elt_var = [(0, 0, 2), (1, 1, 2)]
    val = [1, 0, 0, 1, 1, 0, 3, 5]
    ef = ElementFormat(elt_var, val)
    csr = ef.to_csr()
    print(csr)

    b = [1, 2, 3]
    print(csr.mlp_by_vec(b))
    print(a @ b)
    print()

    a = np.array([[1, 3, 0, 0, 0, 0],
                  [2, 4, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 1, 3],
                  [0, 0, 0, 0, 2, 4]])
    elt_var = [(0, 0, 2), (4, 4, 2)]
    val = [1, 2, 3, 4, 1, 2, 3, 4]
    ef = ElementFormat(elt_var, val)
    csr = ef.to_csr()
    print(csr)

    b = [1, 2, 3, 4, 5, 6]
    print(csr.mlp_by_vec(b))
    print(a @ b)


if __name__ == '__main__':
    main()
