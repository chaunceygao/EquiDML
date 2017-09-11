function D = EuclidDist(galX, probX)
% galX: N1-by-d, probX: N2-by-d
% D: N1-by-N2
    D = bsxfun(@plus, sum(galX.^2, 2), sum(probX.^2, 2)') - 2 * galX * probX';
end
