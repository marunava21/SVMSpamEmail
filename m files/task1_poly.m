function[a, aTest] = task_poly(threshold, Cs,P, xtrain, xtest, trainLabel, testLabel, kerneltype)
for p=1:length(P)
for C=1:length(Cs)
    kp = ((xtrain'*xtrain)+1).^P(p);
    %% Mercer condition check
    gram_m = kp;
    eigenvalues = eig(gram_m);
    tolerance = length(eigenvalues)*eps(max(eigenvalues)); %% custom tolerance checking 
    flag = true;
    if min(eigenvalues) <-tolerance
        flag = false;
        fprintf('P = %d this kernel is not fullfilling the mercer condition\n',P(p))
    end
%% dual problem %%
if flag == true
[row,col] = size(xtrain);
YY = trainLabel*trainLabel';
H = kp.*YY;
f = -ones(col,1);
Aeq = trainLabel';
beq = 0;
lb = zeros(2000,1);
ub = ones(2000,1)*Cs(C);
H = (H+H')/2;
opts = optimset('LargeScale', 'off', 'MaxIter', 1000);
alpha = quadprog(H,f',[],[],Aeq,beq,lb,ub,[],opts); %% solving the quadratic equation
rounded_alpha = find(alpha<threshold); %% finidng the support vectors
alpha(rounded_alpha) = 0;   
% support_vec_idx = find(alpha>0 & alpha<Cs(C));
disp('Quadprog done');
if kerneltype == "poly-Hard"
    idx = find(alpha>0); %% support vector constraint filtration
else
    idx = find(alpha>0 & alpha<=C);%% support vector constraint filtration
end
%% calculation of w and b %%
[weights, bias] = wb(idx, alpha, xtrain, trainLabel, P(p)); %% finding the weight and bias correspond to the support vectors
%% train data %%
a(p,C)=accuracy(alpha, xtrain, xtrain,trainLabel, trainLabel, bias, P(p)); %% calculating the training accuracy
%% test data %%
aTest(p,C) = accuracy(alpha, xtest, xtrain, trainLabel, testLabel, bias, P(p)); %% calculating the test accuracy
end
end
end
end
