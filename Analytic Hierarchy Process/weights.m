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