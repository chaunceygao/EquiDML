%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% Demo code for the following paper:
%%%
%%% Li Zhang, Tao Xiang and Shaogang Gong. 
%%% "Learning a Discriminative Null Space for Person Re-identification". 
%%% Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2016.
%%%
%%% research purpose only. 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% 
clear all;
clc;
close all force;

%% Plot settings
set(0,'DefaultFigureWindowStyle','docked');
colors = repmat('rgbkmcyrgbk',1,200);
markers = repmat('+o*.xsd^v<>ph',1,200);

%% Load data%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
feature_name='GoG'; %LOMO or kCCA

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if strcmp('LOMO',feature_name)
    load E:\wjin\EquiDML_v2.0\mat\viper\viper_lomo.mat  % LOMO feature
    partition = importdata(fullfile('E:\wjin\EquiDML_v2.0\mat\viper', 'partition_random.mat'));
end

if strcmp('kCCA',feature_name)
    load ./data/VIPeR_kcca_split.mat  % KCCA feature
end

if strcmp('GoG', feature_name)
    matDir = 'E:\wjin\RingPush_v1.0\mat\viper\';
    partition = importdata(fullfile('E:\wjin\EquiDML_v2.0\mat\viper', 'partition_random.mat'));
    feat1 = importdata(fullfile(matDir, 'VIPeR_feature_all_GOGyMthetaHSV.mat'));
    feat2 = importdata(fullfile(matDir, 'VIPeR_feature_all_GOGyMthetaLab.mat'));
    feat3 = importdata(fullfile(matDir, 'VIPeR_feature_all_GOGyMthetanRnG.mat'));
    feat4 = importdata(fullfile(matDir, 'VIPeR_feature_all_GOGyMthetaRGB.mat'));
    descriptors = cat(2, feat1, feat2, feat3, feat4);
    descriptors = meanRemovalL2Norm(descriptors');
    descriptors = descriptors';
end

maxNumTemplate = 1;
num_gallery = 316;
num_train = 316;
num_test = 316;

%% Initialize cmc matrix
cmc_null = zeros(num_gallery,3);
cmc_null(:,1) = 1:num_gallery;
nTrial = 10;

FeatP = descriptors(1:632, :)';
FeatG = descriptors(633:end, :)';

for nt=1:nTrial
   disp(['Computing Trial ' num2str(nt) '...' ]);
   
    trnSg = partition(nt).trnSg;
    trnSp = partition(nt).trnSp;
    tstSg = partition(nt).tstSg;
    tstSp = partition(nt).tstSp;

    train_a = FeatP(:, trnSp);
    train_b = FeatG(:, trnSg);
    test_a = FeatP(:, tstSp);
    test_b = FeatG(:, tstSg);
    
%    train_a = trials(nt).featAtrain;
%    train_b = trials(nt).featBtrain;
%    test_a = trials(nt).featAtest;
%    test_b = trials(nt).featBtest;
   
   %% Permutation indices
   idxTrain_a = 1:num_train;
   idxTrain_b = idxTrain_a;
   idxProbe = randperm(num_test);
   idxGallery = randperm(num_gallery);
   
    mm = size(train_a, 2); nn = size(train_b, 2);
    A_label = 1:mm;
    B_label = 1:nn;
  
   %% Permutation on Train, Gallery and Test set
   test_a = test_a(:,idxProbe);
   test_b = test_b(:,idxGallery);
   train_a = train_a(:,idxTrain_a);
   train_b = train_b(:,idxTrain_b);
 
   %% data for NFST
   X = [train_a, train_b]';
   X_labels = [A_label, B_label]';
   Y_a = test_a';
   Y_b = test_b';
   
   %% CHI2 kernel  %best with kCCA feature
   if strcmp('kCCA',feature_name) 
       K = kernel_expchi2(X,X);
       K_a = kernel_expchi2(X, Y_a);
       K_b = kernel_expchi2(X, Y_b);
   end

   %% RBF kernel  %best with LOMO feature   
   if strcmp('LOMO',feature_name)  || strcmp('GoG', feature_name)
       [K, K_a, mu] = RBF_kernel(X, Y_a);
       [K_b] = RBF_kernel2(X, Y_b, mu);
%        K = X * X';
%        K_a = X * Y_a';
%        K_b = X * Y_b';
   end

    %% NFST
    proj_null = NFST(K, X_labels);
    projection_a_null = transpose(K_a)*proj_null;
    projection_b_null = transpose(K_b)*proj_null;
%     score_null = pdist2(projection_b_null, projection_a_null, 'euclidean');
    score_null = EuclidDist(projection_b_null, projection_a_null); % b-vs-a

   %% Compute CMC 
   cmcCurrent = zeros(num_gallery,3);
   cmcCurrent(:,1) = 1:num_gallery;
   for k=1:num_test
       finalScore = score_null(:,k);
       [sortScore sortIndex] = sort(finalScore);
       [cmc_null cmcCurrent] = evaluateCMC_demo(idxProbe(k),idxGallery(sortIndex),cmc_null,cmcCurrent);
   end
%    plotCurrentTrial
   
   fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
   fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', (cmcCurrent([1; 5; 10; 15; 20], 2) ./ cmcCurrent([1;5;10;15;20], 3)) * 100);

%     cmc_tst = EvalCMC( -score_null, 1 : 316, 1 : 316, 100 );
%     fprintf('*****VIPeR: iter %d, testing set cmc *****\n', nt);
%     fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
%     100*cmc_tst(1), 100*cmc_tst(5), 100*cmc_tst(10), 100*cmc_tst(15), 100*cmc_tst(20));
end

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', (cmc_null([1; 5; 10; 15; 20], 2) ./ cmc_null([1;5;10;15;20], 3)) * 100 );


