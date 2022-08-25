function[a] = accuracy(alpha, data1, data2,trainlabel, label, bias, p)
M = size(data1,2);
N = size(data2,2);
for j = 1:M         %
    mat = 0;
    for i = 1:N     %
        mat = mat + alpha(i,:) * trainlabel(i,:) * (data1(:,j)' * data2(:,i) + 1) ^ p;
    end
    gx(j) = mat + bias;
end

for i = 1:M
    if gx(i) > 0
       prediction(i,1) = 1;  
    else
       prediction(i,1) = -1; 
    end
end

a = sum(logical((gx.* label') > 0)) / M * 100;
end