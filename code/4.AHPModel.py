import numpy as np
from itertools import combinations


class AHPModel:

    def __init__(self, field_num):
        '''
        初始化
        :param field_num:指标个数
        '''
        self.field_num = field_num
        self.data = np.identity(self.field_num)

    def _fill_data(self, i, j, value):
        '''
        填充值
        :param i: 第i行,默认从1开始
        :param j: 第j列,默认从1开始
        :param value: 值
        :return:
        '''
        if i <= 0 or j <= 0:
            raise AssertionError('参数行数或列数!不能小于0')
        elif i > self.field_num or j > self.field_num:
            raise AssertionError('当前数据为{}行{}列 请修改参数行数或列数!')

        self.data[i - 1][j - 1] = value

    def add_data(self):
        print('重要程度X:1~10  若A比B弱 则输入倒数:1/X')
        for (i, j) in list(combinations(range(1, self.field_num + 1), 2)):
            value = eval(input('请输入第{}个指标比第{}个指标的重要程度:'.format(i, j)))
            # assert 1 <= value <= 10, '重要程度仅支持1~10分'
            self._fill_data(i, j, value)
            self._fill_data(j, i, 1 / value)

    def get_weight(self):
        self.add_data()
        eig_val, eig_vector = np.linalg.eig(self.data)
        self.max_val = round(np.max(eig_val).real, 4)
        self.check()
        self.vector_list = eig_vector[:, np.argmax(eig_val)].real.round(4)
        vector_sum = sum(self.vector_list)
        weight_vector = ['{:.2%}'.format(vector / vector_sum) for vector in self.vector_list]
        print('{}个指标权重分别为:{}'.format(self.field_num, weight_vector))
        return weight_vector

    # 测试一致性
    def check(self):
        RI_VALUE = [0, 0, 0.58, 0.90, 1.12, 1.24, 1.32, 1.41, 1.45, 1.49]
        # 计算CI值
        CI = (self.max_val - self.field_num) / (self.field_num - 1)
        # 计算CR值
        CR = CI / RI_VALUE[self.field_num]
        CR = round(CR, 4)
        # CR < 0.10才能通过检验
        assert CR < 0.1, '未通过一致性检验 请检查各指标间的重要性程度'

    @classmethod
    def run(cls):
        field_num = int(input('请输入指标的个数:'))
        assert 2 < field_num <= 10, '指标个数要大于2'
        obj = cls(field_num)
        obj.get_weight()


if __name__ == '__main__':
    AHPModel.run()
