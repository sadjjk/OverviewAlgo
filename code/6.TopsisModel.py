import numpy as np


def get_score(data):
    data = np.array(data)
    data_scale = data / data.sum(axis=0)
    Z_add = np.max(data_scale, axis=0)
    Z_sub = np.min(data_scale, axis=0)
    D_add = np.square(Z_add - data_scale).sum(axis=1)
    D_sub = np.square(Z_sub - data_scale).sum(axis=1)
    return list(np.around(D_sub / (D_add + D_sub), 2))

if __name__ == '__main__':
    data = [[67, 90, 67, 81],
            [82, 73, 95, 85],
            [67, 92, 79, 78],
            [82, 81, 86, 76],
            [70, 91, 82, 89],
            [72, 90, 89, 74],
            [82, 85, 92, 69],
            [85, 72, 72, 92],
            [82, 69, 71, 86]]
    get_score(data)
