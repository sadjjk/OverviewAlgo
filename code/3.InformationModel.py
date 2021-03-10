import numpy as np


def get_information_weight(data):
    data = np.array(data)
    cv = data.std(axis=0) / data.mean(axis=0)
    return list(cv / cv.sum())


def get_score(data):
    data = np.array(data)
    weight = np.array(get_information_weight(data))
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
