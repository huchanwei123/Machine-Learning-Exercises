% -----------------------HW3 main function-------------------------
% ---------------------Edited by Chan-Wei Hu-----------------------
clear all;

%% load the data and pre-processing
data = load('./../data/alphabet_DU_training.csv');
test_data = load('./../data/alphabet_DU_testing.csv');
label_col = 1;
% which column is label?
data = data(randperm(size(data,1)), :);
attr_len = length(data(1, :));
% attr_len is total size
label = data(:, label_col);
test_label = test_data(:, label_col);
data(:, label_col) = [];
test_data(:, label_col) = [];
feature = data;
test_feature = test_data;

% transform label to (0,1)
u = unique(label);
u_test = unique(test_label);
label(label==u(1)) = 0;
label(label==u(2)) = 1;
test_label(test_label==u(1)) = 0;
test_label(test_label==u(2)) = 1;

%% Parameter setup here
MAX_ITER = 1000;
C = 3.35;
eps = 0.001;
% eps is convergence criteria, which show what should the difference 
% between lagranage multiplier between two consecutive iteration has to be to exit.
tol = 0.01;
% tol is distance within that the lagrange multiplier will be mapped to zero or upper limit 'C'
type = 'Gaussian';
% choose kernel function for SVM, in this homework, beacuse we don't used
% kernel method, so we just simply dot product.
upper_lower_bound = [0, C];
Sigma = 1;
poly_d = 1;
% sigma for Gaussian Kernel, if you use Linear kernel, just set this to 0
% Actually in this homework, we use 'Linear' only
train_data_Percent = 1;
N = 10;
% N-fold validation

%% Choose the good 'C' which minimize the cross validation error
% In this part, it may cost lots of time to find opyimal C, so just enter a
% C above in parameter 'upper_lower_bound'
%{
C = [];
for i = 1:0.01:10
    C = [C i]
end
R_cv = [];
count = 1;
for i = C
    R_cv = [R_cv estimate_cross_validation(feature, label, eps, tol, type, [0 i], N, Sigma)];
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
        K(i,j)=Kernel(feature(i,:), feature(j,:), type, Sigma, poly_d);
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
            %fprintf('Error: %d\n', Error);
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
out_hypo = [];
for i = 1:data_num
    out_hypo = [out_hypo sign(learned_function(label,lambda,b,i,K))]; 
end
fprintf('Ouptut hypothesis is:\n')
disp(out_hypo);
csvwrite('10-fold_out_hypo.csv', out_hypo);
Train_Error = trainingError(lambda,label,b,K);
%[output_hypothesis, train_accu] = estimate_train_accu(Xback, Yback, lambda, b, kernel_type, data_num)/double(size(Xback, 1) - data_num + 1) * 100;
test_C = test_accu(Xback, Yback, test_feature, test_label, lambda, b, type, Sigma, data_num, poly_d)/size(test_feature, 1) * 100;
%fprintf(sprintf('Training confidence and training error are %d and %d, respectively\n', 100 - C ,Train_Error));
fprintf(sprintf('Testing accuracy is %d percent \n', test_C));
