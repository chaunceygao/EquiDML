function Y = SafeExp(X)
    Y = exp(X);
    Y(isinf(Y)) = realmax;
end
