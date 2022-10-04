%% 优劣解距离法程序
% 加载原始矩阵数据
clear, clc % 初始化
load data_of_river_water_quality.mat

[Excel, n, m] = matPositive(Excel);
Z = matStandard(Excel, n);
weight = entropyWeight(Z, Excel, n, m);
matNormal(Z, n, weight);