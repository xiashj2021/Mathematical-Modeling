```matlab
%% 检验输入的矩阵是否为判断矩阵(符合要求的正互反矩阵)
function [error, n] = check(Mat)
    error = 0; % 默认输入的矩阵是没有问题的
    
    % 检查矩阵是否能构成正常矩阵，是否为方阵
    [row, colum] = size(Mat);
    if row ~= colum || row <= 1
        error = 1;
    end

    % 检查是否为正互反矩阵(a(ij) > 0 && a(ij) * a(ji) = 1)
    if error == 0
        [n, n] = size(Mat);
        if sum(sum(Mat <= 0)) > 0
            error = 2;
        end

        if n > 15
            error = 3;
        end

        if sum(sum(Mat .* Mat' ~= ones(n))) > 0
            error = 4;
        end
   end
end
%                                                                                   check.m

%% 计算最大特征值
function [vector, MaxEig, diagonal] = calculator(Mat)
    [vector, diagonal] = eig(Mat); % 得到矩阵的特征向量和特征值构成的对角矩阵
    MaxEig = max(max(diagonal));
end
%                                                                                   calculator.m

%% 一致性检验
function result = ConsistencyTest(MaxEig, n)
    CI = (MaxEig - n) / (n - 1); % 计算一致性指标CI
    % 平均随机一致性指标表RI
    RI=[0 0.00001 0.52 0.89 1.12 1.26 1.36 1.41 1.46 1.49 1.52 1.54 1.56 1.58 1.59];
    CR = CI / RI(n); % 计算一致性比例
    disp(['一致性指标CI=', num2str(CI)]);
    disp(['一致性比例CR=', num2str(CR)]);

    if CR < 0.10
        disp("该判断矩阵的一致性可以接受");
        result = 0;
    else
        disp("注意!该判断矩阵需要进行修改");
        result = 1;
    end
end
%                                                                                   ConsistencyTest.m

%% 计算权重
function weights(n, vector, MaxEig, Mat, diagonal)
    % 算数平均法
    ColSum = sum(Mat); % 判断矩阵每一列的和
    % 判断矩阵按列归一化
    SumMat = repmat(ColSum, n, 1); % 构造与判断矩阵阶数相同的每列和矩阵
    StandardMat = Mat ./ SumMat; % 求得标准型矩阵

    disp('算术平均法求权重的结果为:');
    disp(sum(StandardMat, 2) ./ n); % 对标准化后的矩阵按照行求和，然后再将这个列向量的每个元素同时除以n

    % 几何平均法
    RowProduct = prod(Mat, 2); % 判断矩阵每一行的积
    ProductMat = RowProduct .^ (1 / n); % 每个分量开n次方
    % 判断矩阵按行归一化
    disp('几何平均法求权重的结果为：');
    disp(ProductMat ./ sum(ProductMat));

    % 特征值法
    [row,column] = find(diagonal == MaxEig, 1); % 找到对角矩阵中第一个与最大特征值相等的元素的位置，记录它的行和列
    disp('特征值法求权重的结果为：');
    % 对最大特征值所在列的特征向量进行归一化
    disp(vector(:, column) ./ sum(vector(:, column)));
end
%                                                                                   weights.m

%% 层次分析法程序
clear; clc % 程序初始化
JudgeMat = input("请输入矩阵:"); % 输入矩阵(n阶方阵，即行数与列数相同)

[error, n] = check(JudgeMat);
if error == 0
    [vector, MaxEig, diagonal] = calculator(JudgeMat);
    result = ConsistencyTest(MaxEig, n);
    if result == 0
        weights(n, vector, MaxEig, JudgeMat, diagonal);
    end
end
%                                                                                   main.m
```
