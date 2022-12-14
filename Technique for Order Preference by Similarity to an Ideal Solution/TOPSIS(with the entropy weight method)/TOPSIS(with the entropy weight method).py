import numpy as np
import pandas as pd


def import_excel_matrix(path):
    data = pd.read_excel(path)
    data = data.values
    return data


# 将输入的数字转化为向量
def get_matrix():
    res = []

    input_line = input()  # 以字符串的形式读入一行
    # 如果不为空字符串作后续读入
    while input_line != '':
        list_line = input_line.split(' ')  # 以空格划分就是序列的形式了
        list_line = [eval(e) for e in list_line]  # 将序列里的数由字符串变为数字类型
        res.append(list_line)

        input_line = input()  # 保持循环

    return res


# 矩阵正向化
def mat_positive(mat):
    [row, column] = np.shape(mat)
    print(f'该矩阵有{row}个评价对象,{column}个评价指标')
    decision = eval(input(f'这{column}个指标是否需要进行正向化处理？需要请输入1，不需要请输入0：'))

    if decision == 1:
        print('请输入需要正向化处理的指标所在的列，例如第2、3、6三列需要处理，那么你需要输入2 3 6后输入两次回车：')
        position_vector = get_matrix()
        print('请输入需要处理的这些列的指标类型（1：极小型，2：中间型，3：区间型）')
        print('例如：第2列是极小型，第3列是区间型，第6列是中间型，就输入1 3 2后输入两次回车：')
        pend_vector = get_matrix()

        for i in range(len(pend_vector[0])):  # 读取待处理向量所在列的个数，从而确定循环次数
            # 极小型指标转化为极大型指标
            if pend_vector[0][i] == 1:
                x = mat[:, position_vector[0][i] - 1]
                print(f'第{position_vector[0][i]}列指标是极小型，正在正向化：')
                x = np.max(x) - x
                print(f'第{position_vector[0][i]}列指标极小型正向化处理完成')
                print('-----------------------------------------------------')
                mat[:, position_vector[0][i] - 1] = x

            # 中间型指标转化为极大型指标
            elif pend_vector[0][i] == 2:
                x = mat[:, position_vector[0][i] - 1]
                print(f'第{position_vector[0][i]}列指标是中间型，正在正向化：')
                best = eval(input('请输入最佳的那一个值：'))
                m = np.max(np.absolute(x - best))
                x = 1 - np.absolute(x - best) / m
                print(f'第{position_vector[0][i]}列指标中间型正向化处理完成')
                print('-----------------------------------------------------')
                mat[:, position_vector[0][i] - 1] = x

            # 区间型指标转化为极大型指标
            elif pend_vector[0][i] == 3:
                x = mat[:, position_vector[0][i] - 1]
                print(f'第{position_vector[0][i]}列指标是区间型，正在正向化：')
                nether = eval(input('请输入区间的下界：'))
                upper = eval(input('请输入区间的上界：'))
                print(f'第{position_vector[0][i]}列指标是区间型，正在正向化：')
                m = np.max([nether - np.min(x), np.max(x) - upper])
                after_process_x = np.ones((row, 1))  # 初始化正向化后的量为1

                for j in range(row):
                    if x[j] < nether:
                        after_process_x[j] = 1 - (nether - x[j]) / m
                    elif x[j] > upper:
                        after_process_x[j] = 1 - (x[j] - upper) / m
                    else:
                        after_process_x[j] = 1
                print(f'第{position_vector[0][i]}列指标区间型正向化处理完成')
                print('-----------------------------------------------------')
                mat[:, position_vector[0][i] - 1] = after_process_x.reshape(row, )

        print('正向化后的矩阵为：\n', mat)
    return [mat, row, column]


# 矩阵标准化
def mat_standard(mat, row):
    z = mat / np.tile((np.sum(mat * mat, axis=0)) ** 0.5, (row, 1))
    print('标准化矩阵为：\n', z)
    return z


# 增加权重
def entropy_weight(z, x, row, column):
    print('请输入是否需要增加权重向量，需要输入1，不需要输入0')
    judge = eval(input('请输入是否需要增加权重：'))
    if judge == 1:
        judge = eval(input('使用熵权法确定权重请输入1，否则输入0：'))
        if judge == 1:
            if np.sum(np.where(z < 0, 1, 0)) > 0:  # 如果之前标准化后的Z矩阵中存在负数，则重新对X进行标准化
                print('原来标准化得到的Z矩阵中存在负数，所以需要对X重新标准化')
                for i in range(row):
                    for j in range(column):
                        z[i, j] = (x[i, j] - np.min(x[:, j])) / (np.max(x[:, j]) - np.min(x[:, j]))
                print('X重新进行标准化得到的标准化矩阵Z为: \n', z)
            [row, column] = np.shape(z)
            d = np.ones((1, column))  # 初始化信息效用值行向量
            for i in range(column):
                temp = z[:, i]  # 取出标准化矩阵的每一列
                p = temp / np.sum(temp, axis=0)
                lnp = np.ones((row, 1))
                for j in range(row):
                    if p[j] == 0:  # 如果第i个元素为0
                        lnp[j] = 0  # 那么返回的第i个结果也为0
                    else:
                        lnp[j] = np.log(p[j])
                else:
                    e = - np.sum(p.reshape(row, 1) * lnp, axis=0) / np.log(row)  # 计算信息熵
                    d[0][i] = 1 - e  # 计算信息效用值
            else:
                weight = d / np.sum(d, axis=1)  # 将信息效用值归一化，得到权重
                print('熵权法确定的权重为：\n', weight)
                return weight
        else:
            print('如果你有3个指标，你就需要输入3个权重，例如它们分别为0.25,0.25,0.5, 则你需要输入0.25 0.25 0.5')
            print(f'你需要输入{column}个权重，请以行向量的形式输入这{column}个权重')
            weight = get_matrix()
            ok = 0  # 用来判断用户的输入格式是否正确
            while ok == 0:
                if (np.absolute(np.sum(weight, axis=1) - 1) < 0.000001) and (np.shape(weight) == [1, column]):
                    ok = 1
                else:
                    print('你输入的有误，请重新输入权重行向量: ')
                    weight = get_matrix()
                    return weight
    else:
        weight = np.ones(1, column) / column  # 如果不需要加权重就默认权重都相同，即都为1/m
        return weight


# 计算得分并归一化
def mat_normal(mat, row, weight):
    d_plus_mat = (np.sum(((mat - np.tile(np.max(mat, axis=0), (row, 1))) ** 2) * np.tile(weight, (row, 1)),
                         axis=1)) ** 0.5  # D+ 与最大值的距离向量
    d_negative_mat = (np.sum(((mat - np.tile(np.min(mat, axis=0), (row, 1))) ** 2) * np.tile(weight, (row, 1)),
                             axis=1)) ** 0.5  # D- 与最小值的距离向量
    s = d_negative_mat / (d_plus_mat + d_negative_mat)  # 未归一化的得分
    standard_s = s / np.sum(s, axis=0)  # 归一化的得分
    print('最后的得分为：\n', standard_s)
    print(f'排序后的值为：{np.sort(standard_s, axis=0)}\n对应的索引为：{np.argsort(standard_s, axis=0)}')


# 优劣解距离法主程序
data_path = 'data_of_river_water_quality.xlsx'
data_matrix = import_excel_matrix(data_path)
print(data_matrix)
[data_matrix, row, column] = mat_positive(data_matrix)
z = mat_standard(data_matrix, row)
weight = entropy_weight(z, data_matrix, row, column)
mat_normal(z, row, weight)
