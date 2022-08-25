function[a,atest] = kernelrbf(threshold, k, C, xtrain, xtest, trainLabel, testLabel, gamma)
%% Mercer condition check %%
gram_m = k;
eigenvalues = eig(gram_m);
flag = true;
tolerance = length(eigenvalues)*eps(max(eigenvalues));%% custom tolerance checking 
if min(eigenvalues) <-tolerance
    flag = false;
    fprintf('this kernel is not fullfilling the mercer condition\n')
end
%% dual problem %%
if flag==true
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
    alpha = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts);%% solving the quadratic equation
    
    rounded_alpha = find(alpha<threshold);%% finidng the support vectors
    alpha(rounded_alpha) = 0;   
    % support_vec_idx = find(alpha>0 & alpha<Cs(C));
    disp('Quadprog done');
    idx = find(alpha>0);%% support vector constraint filtration
%% calculation of w and b %%
    b = zeros(size(idx));
        for j = 1:size(idx)
            s_idx = idx(j);
            weights = 0;
            for i = 1:size(xtrain,2)
                weights = weights + alpha(i,:) * trainLabel(i,:) *  exp((-1 * norm(xtrain(:,s_idx) - xtrain(:,i))) / (10 ^ 2));
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
                mat = mat + alpha(i,:) * trainLabel(i,:) * exp((-1 * norm(xtrain(:,j) - xtrain(:,i))) / (10 ^ 2));
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
    a = sum(prediction==trainLabel) / M * 100; %% train accuracy
%% test data %%
    M = size(xtest,2);
    N = size(xtrain,2);
    for j = 1:M         %
        mat = 0;
        for i = 1:N     %
            mat = mat + alpha(i,:) * trainLabel(i,:) * exp((-1 * norm(xtest(:,j) - xtrain(:,i))) / (10 ^ 2));
        end
        gx1(j) = mat + bias;
    end
    disp(M);
    for i = 1:M
        if gx1(i) > 0 %% if the discriminator value g is +ve then belong to class 1 
           prediction1(i,1) = 1;  
        else %% else class 0
           prediction1(i,1) = -1; 
        end
    end
    
    atest = sum(prediction1==testLabel) / M * 100; %% test accuracy
end
end