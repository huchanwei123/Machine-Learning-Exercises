function out = Kernel(X1, X2, type, Sigma, poly_d)
% Kernel function is used to map the input space to feature space
    if strcmp(type, 'Gaussian') == 1
        out = exp(-norm(X1-X2).^2/(2*Sigma.^2));
    elseif strcmp(type, 'Polynomial') == 1
        out = (1 + sum(X1.*X2)).^poly_d;
    elseif strcmp(type, 'Dot') == 1
        out = sum(X1.*X2);
    else
        error('No matching kernel type !');
    end
end