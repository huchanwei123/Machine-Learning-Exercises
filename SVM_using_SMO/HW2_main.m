% -----------------------HW2 main function-------------------------
% ---------------------Edited by Chan-Wei Hu-----------------------
clear all;

%% load the data
data = load('data/messidor_features_training.csv');
test_data = load('data/messidor_features_testing.csv');
data = data(randperm(size(data,1)), :);
attr_len = length(data(1, :));
% attr_len is feature size
label = data(:, 1);
test_label = test_data(:, 1);
feature = data(:, 2:attr_len);
test_feature = test_data(:, 2:attr_len);

%% Parameter setup here
MAX_ITER = 1000;
eps = 0.001;
% eps is convergence criteria, which show what should the difference 
% between lagranage multiplier between two consecutive iteration has to be to exit.
tol = 0.01;
% tol is distance within that the lagrange multiplier will be mapped to zero or upper limit 'C'
kernel_type = 'Linear';
% choose kernel function for SVM
upper_lower_bound = [0, 1.2];
Sigma = 0;
% sigma for Gaussian Kernel, if you use Linear kernel, just set this to 0
% Actually in this homework, we use 'Linear' only
train_data_Percent = 1;
N = 5;
% N-fold validation

%% Choose the good 'C' which minimize the cross validation error
%{
C = [];
for i = 0.1:0.01:10
    C = [C i]
end
R_cv = [];
count = 1;
for i = C
    R_cv = [R_cv estimate_cross_validation(feature, label, eps, tol, kernel_type, [0 i], N, Sigma)];
    fprintf(sprintf('C = %d , with cross-validation error = %d\n', i, R_cv(count)));
    count = count + 1;
end

[min_error, index] = min(R_cv);
upper_lower_bound = [0 C(index)];
fprintf('Choose the upper bound : %d, with the R_cv = %d \n', C(index), min_error);
%}
%% start SMO
data_num = int32(train_data_Percent * size(feature,1));
test_data_num = size(test_feature,1);
TS = data_num;
lambda = zeros(1, data_num);
Error = zeros(1, data_num);
Xback = feature;
Yback = label;
feature = feature(1:data_num ,:);
label = label(1:data_num);
b = 0;
% Pre-kernel evaluation
K = zeros(size(feature,1));
for i = 1:data_num
    for j = 1:data_num
        K(i,j)=Kernel(feature(i,:), feature(j,:), kernel_type, Sigma);
    end
end

numChanged = 0;
examineAll = 1;
iter = 0;
while((numChanged > 0) || examineAll) && iter < MAX_ITER
    numChanged=0;
    if examineAll
        for i2 = 1:length(lambda)
            [out,lambda,b,Error] = examineExample(i2,label,lambda,tol,upper_lower_bound,Error,eps,K,b);
            numChanged = numChanged + out;
            %fprintf('examineAll %d\n',i2);
        end
    else
        tmp = find(Error > tol & Error < upper_lower_bound(2)-tol);
        for j = 1:length(tmp)
            [out,lambda,b,Error] = examineExample(tmp(j),label,lambda,tol,upper_lower_bound,Error,eps,K,b);
            numChanged = numChanged + out;
            %fprintf('examineNon0NonC %d\n',j);
        end     
    end
    
    if (examineAll == 1)
        examineAll = 0;
    elseif (numChanged == 0)
        examineAll = 1;
    end
    iter = iter + 1;
end
    
Train_Error = trainingError(lambda,label,b,K);
%C = estimate_train_accu(Xback, Yback, lambda, b, kernel_type, data_num)/double(size(Xback, 1) - data_num + 1) * 100;
test_C = test_accu(Xback, Yback, test_feature, test_label, lambda, b, kernel_type, Sigma, data_num)/size(test_feature, 1) * 100;
%fprintf(sprintf('Training confidence and training error are %d and %d, respectively\n', 100 - C ,Train_Error));
fprintf(sprintf('Testing accuracy is %d percent \n', test_C));
