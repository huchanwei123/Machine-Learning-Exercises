% function used to generate bivariate normal distribution.
function [R] = get_bivariate_normal_distribution(mu, sigma, case_)
    if length(mu) ~= 2
        error('mu should be length 2.');
    elseif (size(sigma, 1) ~= 2 & size(sigma, 2) ~= 2)
        error('sigma should be size (2,2).');
    end
    
    if nargin == 3
        R = mvnrnd(mu, sigma, case_);
    else
        R = mvnrnd(mu, sigma);
    end
end