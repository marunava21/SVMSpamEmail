function[weights,bias] = wb(idx, alpha, data,label, p)
b = zeros(size(idx));
weights=0;
for j = 1:size(idx)
    s_idx = idx(j);
    weights = 0;
    for i = 1:size(data, 2)
        weights = weights + alpha(i,:)* label(i,:)* (data(:, s_idx)' * data(:,i)+1)^p;
    end
    b(j) = label(s_idx,:) - weights;
end
bias = mean(b);
end