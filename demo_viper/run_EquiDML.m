function CMC = run_EquiDML(para)

saveresult = para.saveresult;
showresult = para.showresult;
feat_name = para.feat_name;
dataset = para.dataset;
featurefile = para.featurefile;
partitionfile = para.partitionfile;
dim_input = para.dim_input;
tao_neg = para.tao_neg;
tao_pos = para.tao_pos;
total_trial = para.total_trial;
lambda = para.lambda; 
ff = para.ff;
tol = para.tol;
numClass = para.numClass;
II = eye(dim_input);

%================== get and make dirs ============
[demo_dir, filename] = fileparts(mfilename('fullpath'));
exp_dir = fullfile(demo_dir, '..\');
code_dir = fullfile(exp_dir, 'code');
mat_dir = fullfile(exp_dir, ['mat\', dataset]);
res_dir = fullfile(demo_dir, ['\res_', feat_name, '_', filename]);
mkdir(res_dir)
addpath(code_dir);

%================== load feature ===================
partition = importdata(fullfile(mat_dir, partitionfile));

load(fullfile(mat_dir, featurefile)); % 34*1264
%     %========== need normalization? ===========
%     FeatNorm = sum(descriptors.^2, 2);
%     descriptors = sqrt(3)* bsxfun(@rdivide, descriptors, sqrt(FeatNorm));

FeatP = descriptors(1:numClass, :)';
FeatG = descriptors(numClass+1:end, :)'; 

for trial = 1:total_trial
    maxIters = 300;
    L = 1 / 2^8;
    gamma = 2;
    
    info.train.loss = [];
    info.train.rank1 = [];  
    info.val.loss = [];
    info.val.rank1 = [];
    
    trnSg = partition(trial).trnSg;
    trnSp = partition(trial).trnSp;
    tstSg = partition(trial).tstSg;
    tstSp = partition(trial).tstSp;
    
    FeatP_trn_ori = FeatP(:, trnSp);
    FeatG_trn_ori = FeatG(:, trnSg);
    FeatP_tst_ori = FeatP(:, tstSp);
    FeatG_tst_ori = FeatG(:, tstSg);
    FeatTrn_ori = [FeatP_trn_ori, FeatG_trn_ori];
    mu = mean(FeatTrn_ori, 2);
    % [W, ux] = pcaPmtk(FeatTrn_ori', dim_input); % W: ori-by-reduce
    options.ReducedDim = dim_input;
    [W, eigval] = myPCA(FeatTrn_ori', options);

    FeatP_trn = W'*(FeatP_trn_ori - repmat(mu, [1, size(FeatP_trn_ori, 2)]));
    FeatG_trn = W'*(FeatG_trn_ori - repmat(mu, [1, size(FeatG_trn_ori, 2)]));
    FeatP_tst = W'*(FeatP_tst_ori - repmat(mu, [1, size(FeatP_tst_ori, 2)]));
    FeatG_tst = W'*(FeatG_tst_ori - repmat(mu, [1, size(FeatG_tst_ori, 2)]));
    clear FeatP_trn_ori FeatG_trn_ori FeatP_tst_ori FeatG_tst_ori FeatTrn_ori;
    
    % transpose feature matrix
    FeatP_trn = FeatP_trn'; % n-by-d
    FeatG_trn = FeatG_trn';
    FeatP_tst = FeatP_tst';
    FeatG_tst = FeatG_tst';
    
    [nGals, d] = size(FeatG_trn); % n
    nProbs = size(FeatP_trn, 1); % m
    G_label = 1:nGals;
    P_label = 1:nProbs;
    
    % calculate pair label matrix
    Y = bsxfun(@eq, P_label(:), G_label(:)');
    Y = double(Y);
    Y(Y == 0) = -1;
    
    nPos = sum(Y(:) == 1);
    nNeg = sum(Y(:) == -1);
    
    % calculate weight matrix
    nProbs = numel(P_label);
    nGals = numel(G_label);
    
    W = zeros(nProbs, nGals);
    W(Y == 1) = (2-ff) / nPos;
    W(Y == -1) = ff / nNeg;

    % initialize matrix
    M = eye(dim_input); % M_{t-1}
    prevM = eye(dim_input); % M_{t-2}
    prevAlpha = 0;

    eta = zeros(maxIters, 1);
    rankM = zeros(maxIters, 1);
    loss = zeros(maxIters, 1);

    dist_pg = MahDist(M, FeatP_trn, FeatG_trn);

    % calculate loss
    L_neg = (dist_pg - tao_neg).^2; 
    L_neg(Y==1) = 0;
%     sgn = double(dist_pg < tao_neg); L_neg = L_neg.*sgn;
%     L_pos = dist_pg; 
%     L_pos(Y==-1) = 0;
    L_pos = (dist_pg + tao_pos).^2;
    L_pos(Y == -1) = 0;
    Loss = L_neg + L_pos;
    newL = W(:)'*Loss(:) + lambda*(sum(M(:).^2, 1) - sum(II(:).^2, 1));
%     newL = W(:)'*L1(:) + lambda*sum(diag(M), 1);
    
    for iter = 1 : maxIters
        t0 = tic;
        info.train.loss(end+1) = 0;
        info.train.rank1(end+1) = 0;
        info.val.loss(end+1) = 0;
        info.val.rank1(end+1) = 0;
        
        % acceleration
        newAlpha = (1 + sqrt(1 + 4 * prevAlpha^2)) / 2;
        V = M + (prevAlpha - 1) / newAlpha * (M - prevM);
%         V = M;

        prevL = newL;
        prevM = M; 
        prevAlpha = newAlpha;

        dist_pg_Vt = MahDist(V, FeatP_trn, FeatG_trn);

        % calculate gradient
        coef_neg = 2*(dist_pg_Vt - tao_neg);
        coef_neg(Y == 1) = 0;
        coef_pos = 2*(dist_pg_Vt + tao_pos);
        coef_pos(Y == -1) = 0;
        
        coef = coef_neg + coef_pos;
        G = W.*coef;

        G1 = diag(sum(G, 1));
        G2 = diag(sum(G, 2));

        % calculate gradient
        gradL = FeatP_trn'*G2*FeatP_trn - FeatP_trn'*G*FeatG_trn - FeatG_trn'*G'*FeatP_trn + FeatG_trn'*G1*FeatG_trn;
        gradL = gradL + lambda*(V - II);
%         gradL = gradL + lambda*eye(size(V));
        
        % calculate current loss under Vt
        L_neg = (dist_pg_Vt - tao_neg).^2; L_neg(Y==1) = 0;
%         sgn = double(dist_pg_Vt < tao_neg); L_neg = L_neg.*sgn;
%         L_pos = dist_pg_Vt; L_pos(Y==-1) = 0;
        L_pos = (dist_pg_Vt + tao_pos).^2; L_pos(Y==-1) = 0;
        Loss = L_neg + L_pos;
        prevL_V = W(:)'*Loss(:) + lambda*(sum(V(:).^2, 1) - sum(II(:).^2, 1));
%         prevL_V = W(:)'*L1(:) + lambda*sum(diag(V), 1);

        while true
%             [optFlag, M, P, latent, r, newF] = LineSearch(V, gradF, prevF_V, galX, probX, Y, W, L, mu);
            Mt = V - gradL/L;
            Mt = (Mt + Mt')/2;
            [U, S] = eig(Mt);
            latent = max(0, diag(S));
            Mt = U * diag(latent) * U'; % without PSD constraint
            
%             r = sum(latent > 0);
%             [latent, index] = sort(latent, 'descend');
%             latent = latent(1:r);

            % calculate loss under Mt (under current learning step)
            dist_pg_temp = MahDist(Mt, FeatP_trn, FeatG_trn);
            L_neg = (dist_pg_temp - tao_neg).^2; L_neg(Y==1) = 0;
%             sgn = double(dist_pg_temp < tao_neg); L_neg = L_neg.*sgn;
%             L_pos = dist_pg_temp; L_pos(Y==-1) = 0;
            L_pos = (dist_pg_temp + tao_pos).^2; L_pos(Y==-1) = 0;
            Loss = L_neg + L_pos;
            newL = W(:)'*Loss(:) + lambda*(sum(Mt(:).^2, 1) - sum(II(:).^2, 1));
%             newL = W(:)'*L1(:) + lambda*sum(diag(Mt), 1);

            diffM = Mt - V;
            optFlag = newL <= prevL_V + diffM(:)' * gradL(:) + L * norm(diffM, 'fro')^2 / 2;
            if ~optFlag
                L = gamma * L; % to get the optimal learning rate
%                 if verbose == true
                    fprintf('\tEta adapted to %g.\n', 1 / L);
%                 end
            else
                break;
            end
        end
        
        M = Mt;
        
        dist_neg = dist_pg_temp(Y==-1); mu_neg = mean(dist_neg(:));
        dist_pos = dist_pg_temp(Y==1);  mu_pos = mean(dist_pos(:));
        portion = dist_neg > tao_neg; 
        kNeg = sum(portion(:));
        portion = kNeg/nNeg;
        fprintf('===================================    u_neg = %.3f, u_pos = %.3f\n', mu_neg, mu_pos);
        fprintf('===================================    portion = %.3f, kNeg = %d\n', portion, kNeg);
%         eta(iter) = 1 / L;
%         rankM(iter) = r;
%         loss(iter) = newL;

%         if mod(iter, 10) == 0
%             fprintf('Iteration %d: rankM = %d, lossF = %g. Elapsed time: %.3f seconds.\n', iter, rankM(iter), loss(iter), toc(t0));
%         end
% 
        
        %======= Test the CMC performance after eachIteration =======
%         dist_trn = EuclidDist(FeatG_trn*P, FeatP_trn*P);
        dist_trn = MahDist(M, FeatP_trn, FeatG_trn);
        cmc_trn = EvalCMC(-dist_trn, 1:316, 1:316, 100);
        fprintf('*****VIPeR: iter %d, training set cmc *****\n', iter);
        fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
            100*cmc_trn(1), 100*cmc_trn(5), 100*cmc_trn(10), 100*cmc_trn(15), 100*cmc_trn(20));
        
%         dist_tst = EuclidDist(FeatG_tst*P, FeatP_tst*P);
        dist_tst = MahDist(M, FeatP_tst, FeatG_tst);
        cmc_tst{trial} = EvalCMC( -dist_tst, 1 : 316, 1 : 316, 100 );
        fprintf('*****VIPeR: iter %d, testing set cmc *****\n', iter);
        fprintf('rank1\t\trank5\t\trank10\t\trank15\t\trank20 : \n %2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%%\t\t%2.2f%% \n', ...
            100*cmc_tst{trial}(1), 100*cmc_tst{trial}(5), 100*cmc_tst{trial}(10), 100*cmc_tst{trial}(15), 100*cmc_tst{trial}(20));
        
        info.train.loss(end) = newL;
        info.train.rank1(end) = cmc_trn(1);
        info.val.loss(end) = 0;
        info.val.rank1(end) = cmc_tst{trial}(1);
        
        
        % plot the result        
        if showresult == 1
            hh=figure(1);clf;
            subplot(1,2,1);
            semilogy(1:iter, info.train.loss, 'k-'); hold on;
    %         semilogy(1:iter, info.val.loss, 'g-'); hold on;
            xlabel('iteration'); ylabel('loss'); h = legend('train'); grid on;
            set(h, 'color', 'none');
            title('total loss of train');

            subplot(1,2,2);
            plot(1:iter, info.train.rank1, 'k-'); hold on;
            plot(1:iter, info.val.rank1, 'g-');ylim([0 1]);
            xlabel('iteration'); ylabel('cmc rank1'); h = legend('train', 'val'); grid on;
            set(h, 'color', 'none');
            title('error');
            drawnow;
        end
        
        % save result
        if saveresult == 1
            if mod(iter, 20) == 0
                savefig(hh, fullfile(res_dir, [dataset '_' num2str(trial) '.fig']));
                save(fullfile(res_dir, [dataset '_' num2str(iter) '_' num2str(trial) '.mat']), 'cmc_tst');
            end
        end
        
        if (prevL - newL) / prevL < tol
            fprintf('Converged at iter %d. rankM = %d, loss = %g.\n', iter, rankM(iter), loss(iter));
            eta(iter+1 : end) = [];
            rankM(iter+1 : end) = [];
            loss(iter+1 : end) = [];
            save(fullfile(res_dir, [dataset '_final_' num2str(trial) '.mat']), 'cmc_tst', 'M', 'dist_tst');
            break;
        end
        
    end
end

for i=1:total_trial
    name = [dataset '_final_' num2str(i) '.mat'];
    CMC(i,:) = cmc_tst{i};
end