%------------------Machine Learning HW #1--------------------
% EE dept. 
% Name: Chan-Wei Hu
% Student ID: 102061247
%------------------------------------------------------------
clear all;
%----------------------Parameter setup-----------------------
delta = 0.01;
epsolon = 0.1;
mu = [2, 3];
sigma_x = 1;
sigma_y = 3;
corr_coef = 0.5;
sigma = [sigma_x^2, corr_coef*sigma_x*sigma_y; corr_coef*sigma_x*sigma_y, sigma_y^2];
m = ceil((4/epsolon) * log(4/delta));
%------------------------------------------------------------
iteration = ceil(10/delta);
%iteration = 1;
r_test = get_bivariate_normal_distribution(mu, sigma, ceil((19.453/epsolon)^2));
error_larger_than_epsolon = 0;
for i =1:iteration
    r = get_bivariate_normal_distribution(mu, sigma, m);
    concept = check_if_larger_3epsolon(r, epsolon);
    hs = find_hs(concept, mu, sigma, m);
    error = estimate_error(concept, hs, r_test);
    disp('Iteration: ');
    disp(i);
    disp('The Empirical error is: ');
    disp(error);
    if(error > epsolon)
        error_larger_than_epsolon = error_larger_than_epsolon +1;
    end
end
disp('The number of R(hs) larger that epislon is:');
disp(error_larger_than_epsolon);
figure
plot(r(:,1),r(:,2),'+');
hold on
rectangle('Position',[concept(1,1) concept(1,2) concept(2,1)-concept(1,1) concept(2,2)-concept(1,2)],'EdgeColor', 'r');
hold on 
rectangle('Position',[hs(1,1) hs(1,2) hs(2,1)-hs(1,1) hs(2,2)-hs(1,2)],'EdgeColor', 'b');
title('delta = 0.01, epsolon = 0.01');