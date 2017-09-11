%% This is a demo for the MLAPG algorithm, as well as the evaluation on the VIPeR database. You can run this script to reproduce our ICCV 2015 results.
% Note: this demo requires about 1.0-1.5GB of memory.

close all; clear; clc;

feaFile = 'viper_lomo.mat';

pcaDims = -1;

numClass = 632;
numFolds = 10;
numRanks = 100;

%% load the extracted LOMO features
load(feaFile, 'descriptors');
galFea = descriptors(1 : numClass, :);
probFea = descriptors(numClass + 1 : end, :);
clear descriptors

%% set the seed of the random stream. The results reported in our ICCV 2015 paper are achieved by setting seed = 0. 
seed = 0;
rng(seed);

%% evaluate
cms = zeros(numFolds, numRanks);

for nf = 1 : numFolds
    p = randperm(numClass);
    
    galFea1 = galFea( p(1:numClass/2), : );
    probFea1 = probFea( p(1:numClass/2), : );
    
    X = [galFea1; probFea1]; % [n, d]
    mu = mean(X);
    W = PCA(X, pcaDims);
    clear X
    galFea1 = bsxfun(@minus, galFea1, mu) * W;
    probFea1 = bsxfun(@minus, probFea1, mu) * W;
    
    t0 = tic;
    [P, latent, eta, rankM, loss] = MLAPG(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');
    trainTime = toc(t0);

    %{
    %% if you need to set different parameters other than the defaults, set them accordingly
    options.maxIters = 300;
    options.tol = 1e-4;
    options.L = 1 / 2^8;
    options.gamma = 2;
    options.verbose = true;
    [P, latent, eta, rankM, loss] = MLAPG(galFea1, probFea1, (1:numClass/2)', (1:numClass/2)');
    %}
    
    clear galFea1 probFea1
    W = W * P;
    
    galFea2 = galFea(p(numClass/2+1 : end), : );
    probFea2 = probFea(p(numClass/2+1 : end), : );
    galFea2 = bsxfun(@minus, galFea2, mu) * W;
    probFea2 = bsxfun(@minus, probFea2, mu) * W;
    
    t0 = tic;
    dist = EuclidDist(galFea2, probFea2);
    matchTime = toc(t0);
    clear galFea2 probFea2 W P
    
    fprintf('Fold %d: ', nf);
    fprintf('Training time: %.3g seconds. ', trainTime);    
    fprintf('Matching time: %.3g seconds.\n', matchTime);
    
    cms(nf,:) = EvalCMC( -dist, 1 : numClass / 2, 1 : numClass / 2, numRanks );
    clear dist
    
    fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
    fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', cms(nf,[1,5,10,15,20]) * 100);
end

meanCms = mean(cms);
plot(1 : numRanks, meanCms);

fprintf('The average performance:\n');
fprintf(' Rank1,  Rank5, Rank10, Rank15, Rank20\n');
fprintf('%5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%, %5.2f%%\n\n', meanCms([1,5,10,15,20]) * 100);
