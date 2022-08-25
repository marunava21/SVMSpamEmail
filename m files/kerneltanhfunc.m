function[a,atest] = kerneltanh(threshold, k, C, xtrain, xtest, trainLabel, testLabel)
%% Mercer condition check %%
gram_m = k;
eigenvalues = eig(gram_m);
flag = true;
tolerance = length(eigenvalues)*eps(max(eigenvalues));
if min(eigenvalues) <-tolerance
    flag = false;
    fprintf('this kernel is not fullfilling the mercer condition\n')
end
%% dual problem %%
if flag==true
%     k = xtrain'*xtrain;
    % % 
    [row,col] = size(xtrain);
    YY = trainLabel*trainLabel';
    H = k.*YY;
    H=(H+H')/2;
    f = -ones(col,1);
    Aeq = trainLabel';
    beq = 0;
    
    lb = zeros(2000,1);
    ub = ones(2000,1)*C;
    
    opts = optimset('LargeScale', 'off', 'MaxIter', 1000);
    alpha = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts);
    
    rounded_alpha = find(alpha<threshold);
    alpha(rounded_alpha) = 0;   
    % support_vec_idx = find(alpha>0 & alpha<Cs(C));
    disp('Quadprog done');
    idx = find(alpha>0);
%% calculation of w and b %%
    b = zeros(size(idx));
        for j = 1:size(idx)
            s_idx = idx(j);
            weights = 0;
            for i = 1:size(xtrain,2)
                weights = weights + alpha(i,:) * trainLabel(i,:) *  tanh((1/size(xtrain,2))*xtrain(:,s_idx)' * xtrain(:,i)+10);
            end
            b(j) = trainLabel(s_idx,:) - weights;
        end
    bias = mean(b);
%% train data %%
    M = size(xtrain,2);
    N = size(xtrain,2);
    for j = 1:M         %
        mat = 0;
        for i = 1:N     %
            mat = mat + alpha(i,:) * trainLabel(i,:) * tanh((1/size(xtrain,2))*xtrain(:,j)' * xtrain(:,i)+10);
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
    a = sum(logical((gx.* trainLabel') > 0)) / M * 100;
%% test data %%
    M = size(xtest,2);
    N = size(xtrain,2);
    for j = 1:M         %
        mat = 0;
        for i = 1:N     %
            mat = mat + alpha(i,:) * trainLabel(i,:) * tanh((1/size(xtest,2))*xtest(:,j)' * xtrain(:,i)+10);
        end
        gx1(j) = mat + bias;
    end
    disp(M);
    for i = 1:M
        if gx1(i) > 0
           prediction(i,1) = 1;  
        else
           prediction(i,1) = -1; 
        end
    end
    
    atest = sum(logical((gx1.* testLabel') > 0)) / M * 100;
end
end