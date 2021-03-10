import numpy as np
import statsmodels.api as sm
import math


def get_independent_weight(data):
    data = np.array(data)
    R_list = []
    for index in range(data.shape[1]):
        # 计算R值
        X = sm.add_constant(np.delete(data, index, 1))
        y = data[:, index]
        model = sm.OLS(y, X).fit()
        R = round(math.sqrt(model.rsquared), 4)
        R_list.append(R)

    R_1_sum = sum([round(1 / r, 4) for r in R_list])
    return [1 / r / R_1_sum for r in R_list]


def get_score(data):
    data = np.array(data)
    weight = np.array(get_independent_weight(data))
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
