function [K_train, K_test] = Linear_kernel(X, Y)
    K_train = X * X';
    K_test = X * Y';
end