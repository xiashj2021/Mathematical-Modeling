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