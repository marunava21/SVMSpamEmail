function [trD, xtest, trainLabel, testLabel] = dataloader()
clc
clear all
train_data = load("train.mat");
test_data = load("test.mat");
trD=train_data.train_data;
% disp((numel(trD)-nnz(trD))/numel(trD));
trainLabel = train_data.train_label;
xtest = test_data.test_data;
testLabel = test_data.test_label;
%% standardization
trDmean=mean(trD,2);
trDstd = std(trD,0,2);
% trD= (trD - repmat(trDmean,1,2000))./repmat(trDstd,1,2000);
% xtest = (xtest - repmat(trDmean,1,1536))./repmat(trDstd,1,1536);
%% min-max normalization
maximum=(max(trD'))';
minimum = (min(trD'))';
trD=(trD-repmat(minimum, 1, 2000))./(repmat(maximum, 1, 2000)-repmat(minimum, 1, 2000));
xtest =(xtest -repmat(minimum,1,1536))./(repmat(maximum, 1, 1536)-repmat(minimum, 1, 1536));
% disp((numel(trD)-nnz(trD))/numel(trD));
end