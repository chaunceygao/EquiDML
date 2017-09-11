function idx = findIndex(repeated_label, id)
% repeated_label: format: [1 1 2 2 2 3 3 5 5 5 ...]
% id: format: [1 2 5]
% return: idx: find the index of each element of id in repeated_label, keep
% the order unchanged
% for the above example, idx = [1 2 3 4 5 8 9 10]
id = unique(id);