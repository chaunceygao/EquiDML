function dist = BayesianDist(X, Z, A, B, mode)
% X: d-by-n, Z: d-by-m, d is the dimension of feature
% dist_ij = x_i'A'Ax_i/2 + z_j'A'Az_j/2 -x_i'B'Bz_j; 
switch mode
    case 'original'
        AA = A;
        BB = A + B;
    case 'decompose'
        AA = A'*A;
        BB = B'*B;
    otherwise
        error('unknow Bayesian Distance Type');
end
% AA = A'*A;
% BB = B'*B;
u = sum((X'*AA).*X', 2);
v = sum((Z'*AA).*Z', 2);
dist = bsxfun(@plus, u, v')/2 - X'*BB*Z;