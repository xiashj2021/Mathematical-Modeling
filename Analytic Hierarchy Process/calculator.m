%% 计算最大特征值
function [vector, MaxEig, diagonal] = calculator(Mat)
    [vector, diagonal] = eig(Mat); % 得到矩阵的特征向量和特征值构成的对角矩阵
    MaxEig = max(max(diagonal));
end