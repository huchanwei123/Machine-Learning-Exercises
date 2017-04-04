function [R_cv] = estimate_cross_validation(X_, Y_, eps, tol, type, ul, N, Sigma)
% Function used to estimate cross-validation error during choosing the minimizing 
% error of 'C'
    total_R_cv = 0;
    fold_size = int32(size(X_, 1) / N);
    for i = 0:N-1
        if i == N-1
            CV_test = X_(i*fold_size+1:size(X_, 1), :);
            CV_test_label = Y_(i*fold_size+1:size(Y_, 1), :);
        else
            CV_test = X_(i*fold_size+1:(i+1)*fold_size, :);
            CV_test_label = Y_(i*fold_size+1:(i+1)*fold_size, :);
        end
        
        if i == 0
            CV_train = X_((i+1)*fold_size+1:size(X_, 1), :);
            CV_train_label = Y_((i+1)*fold_size+1:size(Y_, 1), :);
        elseif i == N-1
            CV_train = X_(1:i*fold_size+1, :);
            CV_train_label = Y_(1:i*fold_size+1, :);
        else 
            CV_train = [X_(1:i*fold_size+1, :) ;X_((i+1)*fold_size+1:size(X_,1), :)];
            CV_train_label = [Y_(1:i*fold_size+1, :) ;Y_((i+1)*fold_size+1:size(X_,1), :)];
        end
        
        Sdata = size(CV_train, 1);
        % TS = Sdata;
        lambda = zeros(1,Sdata);
        Error = zeros(1,Sdata);
        X = CV_train;
        Y = CV_train_label;
        Xback = X;
        Yback = Y;
        b=0;
        % Pre-kernel evaluation
        K = zeros(size(X,1));
        for i = 1:Sdata
            for j = 1:Sdata
                K(i,j) = Kernel(X(i,:), X(j,:), type, Sigma);
            end
        end

        numChanged=0;
        examineAll=1;

        while((numChanged>0) || examineAll)
            numChanged=0;
            %old=alpha;
            if(examineAll)
                for i2=1:length(lambda)
                    [out,lambda,b,Error]=examineExample(i2,Y,lambda,tol,ul,Error,eps,K,b);
                    numChanged=numChanged+out;
                end
            else
                tmp=find(Error>tol & Error<ul(2)-tol);
                for j=tmp
                    [out,lambda,b,Error]=examineExample(j,Y,lambda,tol,ul,Error,eps,K,b);
                    numChanged=numChanged+out;
                end     
            end
    
            if (examineAll==1)
                examineAll=0;
            elseif (numChanged==0)
                examineAll=1;
            end
        % norm(alpha-old)
        end
        
        R_cv = trainingError(lambda,Y,b,K);
        total_R_cv = total_R_cv + R_cv;
        test_C = test_accu(Xback, Yback, CV_test, CV_test_label, lambda, b, type, Sigma, Sdata)/size(CV_test, 1) * 100;
        fprintf(sprintf('[Cross-Validation] Confidence and cross-validation error are %d and %d\n', test_C , R_cv));
    end
    R_cv = total_R_cv / N;
end