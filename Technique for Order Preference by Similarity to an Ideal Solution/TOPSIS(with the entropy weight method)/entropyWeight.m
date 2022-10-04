%% 增加权重
function [weight] = entropyWeight(Z, X, n, m)
    disp("请输入是否需要增加权重向量，需要输入1，不需要输入0");
    Judge = input('请输入是否需要增加权重：');
    if Judge == 1
        Judge = input('使用熵权法确定权重请输入1，否则输入0：');
        if Judge == 1
            if sum(sum(Z<0)) >0   % 如果之前标准化后的Z矩阵中存在负数，则重新对X进行标准化
                disp('原来标准化得到的Z矩阵中存在负数，所以需要对X重新标准化')
                for i = 1:n
                    for j = 1:m
                        Z(i,j) = [X(i,j) - min(X(:,j))] / [max(X(:,j)) - min(X(:,j))];
                    end
                end
                disp('X重新进行标准化得到的标准化矩阵Z为: ')
                disp(Z)
            end    
            [n, m] = size(Z);
            d = ones(1, m); % 初始化信息效用值行向量
            for i = 1 : m
                x = Z(:, i); % 取出标准化矩阵的每一列
                p = x / sum(x);
                lnp = ones(n, 1);
                for j = 1 : n
                    if p(j) == 0   % 如果第i个元素为0
                    lnp(j) = 0;  % 那么返回的第i个结果也为0
                    else
                    lnp(j) = log(p(j));  
                    end
                end
                e = - sum(p .* lnp) / log(n); % 计算信息熵
                d(i) = 1 - e; % 计算信息效用值
            end
            weight = d ./ sum(d);  % 将信息效用值归一化，得到权重
            disp('熵权法确定的权重为：');
            disp(weight);
        else
            disp(['如果你有3个指标，你就需要输入3个权重，例如它们分别为0.25,0.25,0.5, 则你需要输入[0.25,0.25,0.5]']);
            weight = input(['你需要输入' num2str(m) '个权数。' '请以行向量的形式输入这' num2str(m) '个权重: ']);
            OK = 0;  % 用来判断用户的输入格式是否正确
            while OK == 0 
                if abs(sum(weight) -1)<0.000001 && size(weight,1) == 1 && size(weight,2) == m
                    OK =1;
                else
                    weight = input('你输入的有误，请重新输入权重行向量: ');
                end
            end
        end
    else
        weight = ones(1,m) ./ m ; %如果不需要加权重就默认权重都相同，即都为1/m
    end
end