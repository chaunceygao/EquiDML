function Y = Logistic(X)
    Y = log(1 + exp(X));
    Y(isinf(Y)) = X(isinf(Y));
end