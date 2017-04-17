function [accu] = test_accu(X, Y, test_data, test_label, lambda, b, type, Sigma, Sdata, d_poly)
% Evaluate Kernel function

    K = zeros(size(test_data,1),Sdata);
    
    for i = 1:size(test_data,1)
        for j = 1:Sdata
            K(i,j) = Kernel(test_data(i,:),X(j,:),type,Sigma, d_poly);
        end
    end
    
    P=0.0;
    for i = 1:size(test_data,1)
        out = sum(Y(1:Sdata)'.*lambda.*K(i,:)) - b;
        out
        if (out < 0 && test_label(i) ~= 0)
            P = P + 1;
        elseif (out > 0 && test_label(i) ~= 1)
            P = P + 1;
        end
    end  
    accu = size(test_data, 1) - P;
end