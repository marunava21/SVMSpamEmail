
function[a,atest] = tasklinearhard(threshold, k, C, xtrain, xtest, trainLabel, testLabel)
%% Mercer condition check %%
gram_m = k;
eigenvalues = eig(gram_m);
tolerance = length(eigenvalues)*eps(max(eigenvalues));%% custom tolerance checking 
flag = true;
if min(eigenvalues) <-tolerance
    flag = false;
    fprintf('this kernel is not fullfilling the mercer condition\n')
end
%% dual problem %%
if flag==true
    k = xtrain'*xtrain; %% linear kernel
    % % 
    [row,col] = size(xtrain);
    YY = trainLabel*trainLabel';
    H = k.*YY;
    f = -1*ones(2000,1);
    Aeq = trainLabel.';
    beq = 0;
    
    lb = zeros(2000,1);
    ub = C.*ones(2000,1);
    
    opts = optimset('LargeScale', 'off', 'MaxIter', 1000);
    alpha = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts);
    
    rounded_alpha = find(alpha<threshold);
    alpha(rounded_alpha) = 0;   
    disp('Quadprog done');
    idx = find(alpha>0);
%% finding w and b %%
    weights = sum(alpha'.*trainLabel'.*xtrain,2);
    b = (1./trainLabel(idx) - xtrain(:, idx)'*weights);
    bias = mean(b);
%% train data %%
    M = size(xtrain,2);
    N = size(xtrain,2);
    for j = 1:M         
        mat = 0;
        for i = 1:N     
            mat = mat + alpha(i,:) * trainLabel(i,:) * (xtrain(:,j)' * xtrain(:,i) );
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
    for j = 1:M         
        mat = 0;
        for i = 1:N     
            mat = mat + alpha(i,:) * trainLabel(i,:) * (xtest(:,j)' * xtrain(:,i));
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
    disp(sum(prediction1==testLabel))
    atest = sum(prediction1==testLabel) / M * 100; %% test accuracy
end
end
