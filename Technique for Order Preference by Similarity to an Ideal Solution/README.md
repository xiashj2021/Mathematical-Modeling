```matlab
% MATLAB
%% 矩阵正向化
function [mat, n] = matPositive(mat)
    [n, m] = size(mat);
    disp(['该矩阵有' num2str(n) '个评价对象', num2str(m) '个评价指标']);
    decision = input(['这' num2str(m) '个指标是否需要进行正向化处理？需要请输入1，不需要请输入0：']);

    if decision == 1
        positionVector = input('请输入需要正向化处理的指标所在的列，例如第2、3、6三列需要处理，那么你需要输入[2,3,6]： ');
        disp('请输入需要处理的这些列的指标类型（1：极小型，2：中间型，3：区间型）');
        pendVector = input('例如：第2列是极小型，第3列是区间型，第6列是中间型，就输入[1,3,2]：');

        for i = 1 : size(positionVector, 2) % 读取待处理向量所在列的个数，从而确定循环次数
            % 极小型指标转化为极大型指标
            if pendVector(i) == 1
                X = mat(:, positionVector(i));
                disp(['第' num2str(positionVector(i)) '列指标是极小型，正在正向化：']);
                X = max(X) - X;
                disp(['第' num2str(positionVector(i)) '列指标极小型正向化处理完成']);
                disp('-----------------------------------------------------');
                mat(:, positionVector(i)) = X;
            % 中间型指标转化为极大型指标
            elseif pendVector(i) == 2
                X = mat(:, positionVector(i));
                disp(['第' num2str(positionVector(i)) '列指标是中间型，正在正向化：']);
                best = input('请输入最佳的那一个值：');
                M = max(abs(X-best));
                X = 1 - abs(X-best) / M;
                mat(:, positionVector(i)) = X;
                disp(['第' num2str(positionVector(i)) '列指标中间型正向化处理完成']);
                disp('-----------------------------------------------------');
            % 区间型指标转化为极大型指标
            elseif pendVector(i) == 3
                X = mat(:, positionVector(i));
                nether = input('请输入区间的下界： ');
                upper = input('请输入区间的上界： '); 
                disp(['第' num2str(positionVector(i)) '列指标是区间型，正在正向化：']);
                row = size(X, 1); % 获取原始矩阵中需要正向化的列的行数
                M = max([nether-min(x), max(x)-upper]);
                afterProcessX = ones(row, 1); % 初始化正向化后的量为1

                for j = 1 : row
                    if X(j) < nether
                        afterProcessX(j) = 1 - (nether - X(j)) / M;
                    elseif X(i) > upper
                        afterProcessX(j) = 1 - (X(j) - upper) / M;
                    else
                        afterProcessX(j) = 1;
                    end
                end
                mat(:, positionVector(i)) = afterProcessX;
                disp(['第' num2str(positionVector(i)) '列指标区间型正向化处理完成']);
                disp('-----------------------------------------------------');
            else
                disp('没有这种类型的指标，请检查指标类型中是否输入了1、2、3之外的其他值');
            end
        end
        disp('正向化后的矩阵为：');
        disp(mat);
    end
end
%                                                                                        matPositive.m

%% 矩阵标准化
function [Z] = matStandard(mat, n)
    Z = mat ./ repmat(sum(mat.*mat) .^ 0.5, n, 1);
    disp('标准化矩阵为：');
    disp(Z);
end
%                                                                                        matStandard.m

%% 计算得分并归一化
function matNormal(mat, n)
    Dplus = sum((mat - repmat(max(mat), n, 1)) .^ 2, 2) .^ 0.5;   % D+即与最大值的距离向量
    Dnegative = sum((mat - repmat(min(mat),n,1)) .^ 2, 2) .^ 0.5;   % D-即与最小值的距离向量
    S = Dnegative ./ (Dplus + Dnegative);    % 未归一化的得分
    disp('最后的得分为：');
    standMat = S / sum(S) % 向量归一化
    [sortedMat, index] = sort(standMat ,'descend') % 向量按降序排列
end
%                                                                                        matNormal.m

%% 优劣解距离法程序
% 加载原始矩阵数据
clear, clc % 初始化
load data_of_river_water_quality.mat

[Excel, n] = matPositive(Excel);
Z = matStandard(Excel, n);
matNormal(Z, n);
%                                                                                        main.m

```


```python3
# python
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
                mat[:, position_vector[0][i] - 1] = after_process_x.reshape(row,)

        print('正向化后的矩阵为：\n', mat)
    return [mat, row]


# 矩阵标准化
def mat_standard(mat, row):
    z = mat / np.tile((np.sum(mat * mat, axis=0)) ** 0.5, (row, 1))
    print('标准化矩阵为：\n', z)
    return z


# 计算得分并归一化
def mat_normal(mat, row):
    d_plus_mat = (np.sum((mat - np.tile(np.max(mat, axis=0), (row, 1))) ** 2, axis=1)) ** 0.5  # D+ 与最大值的距离向量
    d_negative_mat = (np.sum((mat - np.tile(np.min(mat, axis=0), (row, 1))) ** 2, axis=1)) ** 0.5  # D- 与最小值的距离向量
    s = d_negative_mat / (d_plus_mat + d_negative_mat)  # 未归一化的得分
    standard_s = s / np.sum(s, axis=0)  # 归一化的得分
    print('最后的得分为：\n', standard_s)
    print(f'排序后的值为：{np.sort(standard_s, axis=0)}\n对应的索引为：{np.argsort(standard_s, axis=0)}')


# 优劣解距离法主程序
data_path = 'data_of_river_water_quality.xlsx'
data_matrix = import_excel_matrix(data_path)
print(data_matrix)
[data_matrix, row] = mat_positive(data_matrix)
z = mat_standard(data_matrix, row)
mat_normal(z, row)


```
