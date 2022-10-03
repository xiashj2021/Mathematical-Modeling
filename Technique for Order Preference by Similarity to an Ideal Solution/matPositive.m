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
                M = max(abs(X)-best);
                X = 1 - abs(X-best) / M;
                mat(:, positionVector(i)) = X;
                disp(['第' num2str(positionVector(i)) '列指标中间型正向化处理完成']);
                disp('-----------------------------------------------------');
            % 区间型指标转化为极大型指标
            elseif pendVector(i) == 3
                X = mat(:, positionVector(i));
                disp(['第' num2str(positionVector(i)) '列指标是区间型，正在正向化：']);
                nether = input('请输入区间的下界： ');
                upper = input('请输入区间的上界： '); 
                disp(['第' num2str(positionVector(i)) '列指标是区间型，正在正向化：']);
                row = size(X, 1); % 获取原始矩阵中需要正向化的列的行数
                M = max([nether-min(X), max(X)-upper]);
                afterProcessX = ones(row, 1); % 初始化正向化后的量为1

                for j = 1 : row
                    if X(j) < nether
                        afterProcessX(j) = 1 - (nether - X(j)) / M;
                    elseif X(j) > upper
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