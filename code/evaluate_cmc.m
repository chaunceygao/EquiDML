function cmc = evaluate_cmc(sim)
% sim±ÿ–Î «probe-by-gallery
[num_probe, num_gallery] = size(sim);
label = (1:num_probe)';
[~, idx] = sort(sim, 2, 'descend');
label_mat = repmat(label, [1, num_gallery]);
result = (idx == label_mat);
vec = sum(result, 1);
cmc = cumsum(vec);
end