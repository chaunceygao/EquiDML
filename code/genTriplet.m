function tripletIndex = genTriplet(id_batch)
% generate triplet index according to the input id
% input:
% id_batch: [1, n], each element indicate the id of the corresponding image
% in im_batch and feature_map
% output:
% tripletIndex: [N,3], N is the total number of triplets
% triplet:(x, x_pos, x_neg)
tripletIndex = [];
for i=1:numel(id_batch)
    index_prb = i;
    index_pos = setdiff(find(id_batch == id_batch(index_prb)), index_prb);
    index_neg = setdiff(1:numel(id_batch), [index_prb, index_pos]);
    
    [X, Y, Z] = meshgrid(index_prb, index_pos, index_neg);
    tripletIndex = cat(1, tripletIndex, [X(:), Y(:), Z(:)]);
end