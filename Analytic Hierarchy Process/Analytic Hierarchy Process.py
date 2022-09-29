import numpy as np


# 检验输入的矩阵是否为判断矩阵(符合要求的正互反矩阵)
def check(mat):
    error = 0  # 默认输入的矩阵是没有问题的

    # 检查矩阵是否能构成正常矩阵，是否为方阵
    [row, column] = np.shape(mat)
    if row != column or row <= 1:
        error = 1

    # 检查是否为正互反矩阵(a(ij) > 0 & & a(ij) * a(ji) = 1)
    if error == 0:
        [n, n] = np.shape(mat)
        if np.sum(np.where(mat <= 0, 1, 0)) > 0:
            error = 2

        if n > 15:
            error = 3

        if np.sum(np.where(np.multiply(mat.T, mat) != 1, 1, 0)) > 0:
            error = 4

        return [error, n]


# 计算最大特征值
def calculator(mat):
    lamda = np.linalg.eig(mat)  # 求解特征值及特征向量

    index = np.argmax(lamda[0])
    eig_max = np.real(lamda[0][index])
    vector = lamda[1][:, index]

    vector_final = np.real(vector)
    return [vector_final, eig_max]


# 一致性检验
def consistency_test(eig_max, n):
    ci = (eig_max - n) / (n - 1)  # 计算一致性指标CI
    # 平均随机一致性指标表RI
    ri = [0, 0.00001, 0.52, 0.89, 1.12, 1.26, 1.36, 1.41, 1.46, 1.49, 1.52, 1.54, 1.56, 1.58, 1.59]
    cr = ci / ri[n - 1]  # 计算一致性比例CR
    print("一致性指标CI = ", ci)
    print("一致性比例CR = ", cr)

    if cr < 0.10:
        print("该判断矩阵的一致性可以接受")
        return 0
    else:
        print("注意!该判断矩阵需要进行修改")
        return 1


# 计算权重
def weights(n, vector_final, eig_max, mat):
    # 算数平均法
    col_sum = np.sum(mat, axis=0)  # 判断矩阵每一列的和
    # 判断矩阵按列归一化
    sum_mat = np.tile(col_sum, (n, 1))  # 构造与判断矩阵阶数相同的每列和矩阵
    standard_mat = mat / sum_mat  # 求得标准型矩阵

    print('算术平均法求权重的结果为:')
    print(np.sum(standard_mat, axis=1) / n)  # 对标准化后的矩阵按照行求和，然后再将这个列向量的每个元素同时除以n

    # 几何平均法
    row_product = np.prod(mat, axis=1)  # 判断矩阵每一行的积
    product_mat = row_product ** (1 / n)  # 每个分量开n次方
    # 判断矩阵按行归一化
    print('几何平均法求权重的结果为：')
    print(product_mat / np.sum(product_mat))

    # 特征值法
    print('特征值法求权重的结果为：')
    # 对最大特征值所在列的特征向量进行归一化
    print(vector_final / np.sum(vector_final))

# 层次分析法主程序
# 读取文件
'''
txt_path = 'H:\Data\Project\Pycharm\data.txt'  # txt文本路径
f = open(txt_path)
data_lines = f.readlines()
dataset = []
for data in data_lines:
    data1 = data.strip("\n")
    data2 = data1.split("\t")
    dataset.append(data2)

judge_mat = np.array(dataset)
'''
judge_mat = np.array([[1, 1, 4, 1/3, 3],
                      [1, 1, 4, 1/3, 3],
                      [1/4, 1/4, 1, 1/3, 1/2],
                      [3, 3, 3, 1, 3],
                      [1/3, 1/3, 2, 1/3, 1]])
[error, n] = check(judge_mat)
if error == 0:
    [vector_final, eig_max] = calculator(judge_mat)
    result = consistency_test(eig_max, n)
    if result == 0:
        weights(n, vector_final, eig_max, judge_mat)
elif error == 1:
    print("请检查矩阵的维数是否不大于1或不是方阵")
elif error == 2:
    print("请检查矩阵中有元素小于等于0")
elif error == 3:
    print("矩阵的维数n超过了15，请减少准则层的数量")
elif error == 4:
    print("该矩阵不为正互反矩阵")
