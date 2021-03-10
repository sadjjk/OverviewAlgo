import numpy as np


def get_critic_weight(data):
    data = np.array(data)
    std = data.std(axis=0, ddof=1)
    R = (1 - np.corrcoef(data.T)).sum(axis=0)
    C = std * R
    return list(C / C.sum())


def get_score(data):
    data = np.array(data)
    weight = np.array(get_critic_weight(data))
    return list(np.around((data * weight).sum(axis=1), 2))


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
