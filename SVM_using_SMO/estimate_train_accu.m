function [P] = estimate_train_accu(X, Y, lambda, b, type, Sdata)
% Evaluate Kernel function
    K = zeros(size(X,1)-Sdata+1,Sdata);
    for i = Sdata+1:size(X,1)
        for j = 1:Sdata
            K(i-Sdata,j) = Kernel(X(i,:),X(j,:),type);
        end
    end
    
    P=0.0;
    for i = Sdata+1:size(X,1)
        out = sum(Y(1:Sdata)'.*lambda.*K(i-Sdata,:)) - b;
        if ((out*Y(i)) < 0)
            P = P + 1;
        end
    end  
end