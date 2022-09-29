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
elseif error == 1
    disp("请检查矩阵的维数是否不大于1或不是方阵")
elseif error == 2
    disp("请检查矩阵中有元素小于等于0")
elseif error == 3
    disp("矩阵的维数n超过了15，请减少准则层的数量")
elseif error == 4
    disp("该矩阵不为正互反矩阵")
end