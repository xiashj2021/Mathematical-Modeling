%% 矩阵标准化
function [Z] = matStandard(mat, n)
    Z = mat ./ repmat(sum(mat.*mat) .^ 0.5, n, 1);
    disp('标准化矩阵为：');
    disp(Z);
end