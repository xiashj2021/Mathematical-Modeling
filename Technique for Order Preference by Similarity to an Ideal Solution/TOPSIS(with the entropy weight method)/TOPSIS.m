%% ���ӽ���뷨����
% ����ԭʼ��������
clear, clc % ��ʼ��
load data_of_river_water_quality.mat

[Excel, n, m] = matPositive(Excel);
Z = matStandard(Excel, n);
weight = entropyWeight(Z, Excel, n, m);
matNormal(Z, n, weight);