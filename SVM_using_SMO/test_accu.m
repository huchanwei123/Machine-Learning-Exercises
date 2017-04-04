function [accu] = test_accu(X, Y, test_data, test_label, lambda, b, type, Sigma, Sdata)
% Evaluate Kernel function

    K = zeros(size(test_data,1),Sdata);
    for i = 1:size(test_data,1)
        for j = 1:Sdata
            K(i,j) = Kernel(test_data(i,:),X(j,:),type,Sigma);
        end
    end

    P=0.0;
    for i = 1:size(test_data,1)
        out = sum(Y(1:Sdata)'.*lambda.*K(i,:)) - b;
        if (test_label(i)*out < 0)
            P = P + 1;
        end
    end  
    accu = size(test_data, 1) - P;
end