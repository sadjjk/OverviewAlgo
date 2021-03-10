import numpy as np


def get_entropy_weight(data):
    '''
    计算权重系数
    :param data: 二维数组 n行*m列 n个待评价对象 m个评价指标
    :return:指标权重 一维数组 m个值
    '''
    data = np.array(data)
    # 归一化
    P = data / data.sum(axis=0)
    # 计算熵值
    E = np.nansum(-P * np.log(P) / np.log(len(data)), axis=0)
    # 权重系数
    return list((1 - E) / (1 - E).sum())


def get_score(data):
    '''
    计算对象得分
    :param data: 二维数组 n行*m列 n个待评价对象 m个评价指标
    :return:
    '''

    data = np.array(data)
    weight = np.array(get_entropy_weight(data))
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
