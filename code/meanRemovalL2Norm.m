function featout = meanRemovalL2Norm(featin)
% featin: D-by-N, featout: D-by-N
meanx = mean(featin, 2);
% mean removal
feat = featin - repmat(meanx, [1, size(featin, 2)]);
mag = sqrt(sum(feat.^2, 1));
featout = bsxfun(@rdivide, feat, mag);