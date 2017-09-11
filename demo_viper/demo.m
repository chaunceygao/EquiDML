%在Demo_v2中，我们不使用LineSearch来调节学习率，而是使用固定学习率，看看收敛效果有什么不同

clear; clc;

%================== parameters for viper ===============
para.showresult = 0;
para.saveresult = 0;

para.feat_name = 'LOMO';
para.dataset = 'viper';
para.dim_input = 631;
para.featurefile = 'viper_lomo.mat';  %  you can calculate other types of features, and change the file name here
                                      %  this feature file is obtained using LOMO code, which has been given in '..\code\LOMO_XQDA'
para.partitionfile = 'partition_random.mat';

para.tao_neg = 2;
para.tao_pos = 1;
para.total_trial = 10;
para.lambda = 2e-4; % 2e-4 for LOMO, 1e-4 for GoG
para.ff = 1;
para.tol = 1e-5;
para.numClass = 632;

if strcmp(para.feat_name, 'LOMO')
    para.lambda = 2e-4;
end
if strcmp(para.feat_name, 'GoG')
    para.lambda = 1e-4;
end

% perform re-id
CMC = run_EquiDML(para);

% performance evaluation
cal_CMC(CMC)