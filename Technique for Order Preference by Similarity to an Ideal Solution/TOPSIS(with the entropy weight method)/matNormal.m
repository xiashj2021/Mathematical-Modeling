%% 计算得分并归一化
function matNormal(mat, n, weight)
    % Dplus = sum((mat - repmat(max(mat), n, 1)) .^ 2, 2) .^ 0.5;   % D+ 与最大值的距离向量
    % Dnegative = sum((mat - repmat(min(mat), n, 1)) .^ 2, 2) .^ 0.5;   % D- 与最小值的距离向量
    Dplus = sum([(mat - repmat(max(mat), n, 1)) .^ 2] .* repmat(weight, n, 1), 2) .^ 0.5;   % D+ 与最大值的距离向量
    Dnegative = sum([(mat - repmat(min(mat), n, 1)) .^ 2] .* repmat(weight, n, 1), 2) .^ 0.5;   % D- 与最小值的距离向量
    S = Dnegative ./ (Dplus + Dnegative);    % 未归一化的得分
    disp('最后的得分为：');
    standardMat = S / sum(S)
    [sortedMat,index] = sort(standardMat ,'descend')
end