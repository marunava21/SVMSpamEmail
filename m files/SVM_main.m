clc
clear all
[xtrain, xtest, trainLabel, testLabel] = dataloader(); %% data loading 
prompt ="what type of kernel? such as [Linear-Hard, poly-Hard, poly-soft, tanh]\n";
x=input(prompt,"s");
prompt1="threshold such as 1e-6 or 1e-4 or 1e5\n";
y=input(prompt1);
[a, atest] = eval_func(xtrain, xtest, trainLabel, testLabel, x, y) %% main function



%% function definition
function[a, atest] = eval_func(xtrain, xtest, trainLabel, testLabel, kerneltype, y)
if kerneltype=="linearhard"
    threshold = y;
    C = 1e6; 
    k = xtrain'*xtrain;
    [a, atest] = task_linear_hard(threshold, k, C, xtrain, xtest, trainLabel, testLabel);
elseif kerneltype == "polyhard"
    threshold = y;
    Cs= [10^6];
    P = [1,2,3,4,5];
    [a, atest] = task1_poly(threshold, Cs,P, xtrain, xtest, trainLabel, testLabel, kerneltype);
elseif kerneltype == "polysoft"
    threshold = y;
    Cs= [0.1, 0.6, 1.1, 2.1];
    P = [1,2,3,4,5];
    [a, atest] = task1_poly(threshold, Cs,P, xtrain, xtest, trainLabel, testLabel, kerneltype);
elseif kerneltype == "tanh"
    threshold = y;
    C = 10^6; 
    k = tanh((1/size(xtrain,2))*xtrain'*xtrain+10);
    [a, atest] = kerneltanhfunc(threshold, k, C, xtrain, xtest, trainLabel, testLabel);
elseif kerneltype =="rbf"
    threshold = y;
    C=10^6;
    gamma=10^-3;
    k=zeros(2000,2000);
    for i=1:size(xtrain,2)
        for j=1:size(xtrain,2)
            k(i,j)=exp((-1 * norm(xtrain(:,j) - xtrain(:,i))) / (10 ^ 2));
        end
    end
    [a, atest] = kernelrbffunc(threshold, k, C, xtrain, xtest, trainLabel, testLabel,gamma);
end
end

