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