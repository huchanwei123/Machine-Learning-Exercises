function [out, lambda, Error, b] = takeStep(i1, i2, lambda, Y, b, eps, Error, ul, K, E2, tol)
% eps is the convergence on alpha vector elements, default is 0.001
% tol is the distance from lower '0' and upper 'C' bounds, default=0.001
% ul is the upper and lower bound values [ul(1), ul(2)]=[lower, upper]
% K is the kernel matrix evaluated once
    out = 0;
    if(i1 == i2)
        return;
    end
    
    % lagrange multiplier for i1 and i2
    lambda1 = lambda(i1);
    lambda2 = lambda(i2);
    
    % check if alpha1 is on-bound --> [YES; NO] = [Evaluate E1 from SVM function; 
    % Take E1 from cache Error vector]
    if ((lambda1 < tol) || (lambda1 > ul(2)-tol))
        E1 = learned_function(Y,lambda,b,i1,K) - Y(i1);
    else
        E1 = Error(i1);
    end
    
    % Computing the linear constraint bound
    s = Y(i1)*Y(i2);
    
    if(s > 0)
        % if target y1 equals the target y2
        L = max(0, lambda1 + lambda2 - ul(2));
        H = min(ul(2), lambda1 + lambda2);
    else
        % else if y1 does not equal to y2
        L = max(0, lambda2 - lambda1);
        H = min(ul(2), ul(2) + lambda2 - lambda1);
    end
    
    if (L == H)
        % check if the bound equal or not
        return;
    end
    
    k11 = K(i1,i1);
    k12 = K(i1,i2);
    k22 = K(i2,i2);
    eta = k11 + k22 - 2 * k12; 
    gamma = lambda(i1) + s * lambda(i2);
    
    if(eta > 0)
    % Under normal circumstances, and eta will be less than zero. In this case, SMO computes the 
    % maximum along the direction of the constraint:
        a2 = lambda2 + Y(i2) * (E1 - E2) / eta;
        % a2 is new lambda2, and lambda2 is old one.
        if a2 < L
            a2 = L;
        elseif a2 > H
            a2 = H;
        end
    else
    % SMO will work even when is negative, in which case the objective function W should be evaluated 
    % at each end of the line segment
        Lobj=-s*L+L-0.5*k11*(gamma-s*L)^2-.5*k22*L^2-s*k12*(gamma-s*L)*L-Y(i1)...
            *(gamma-s*L)*(learned_function(Y,lambda,b,i1,K)+b-Y(i1)*lambda1*k11-Y(i2)*lambda2*K(i2,i1))...
            -Y(2)*L*(learned_function(Y,lambda,b,i2,K)+b-Y(i1)*lambda1*k12-Y(i2)*lambda2*k22);
     
        Hobj=-s*H+H-0.5*k11*(gamma-s*H)^2-.5*k22*H^2-s*k12*(gamma-s*H)*H-Y(i1)...
            *(gamma-s*H)*(learned_function(Y,lambda,b,i1,K)+b-Y(i1)*lambda1*k11-Y(i2)*lambda2*K(i2,i1))...
            -Y(2)*H*(learned_function(Y,lambda,b,i2,K)+b-Y(i1)*lambda1*k12-Y(i2)*lambda2*K(i2,i1));
        
        if Lobj < Hobj-eps
            a2 = L;
        elseif Lobj > Hobj+eps
            a2 = H;
        else
            a2 = lambda2;
        end
        %fprintf('eta < 0');
    end
    
    if(a2 < 1e-8)
        a2 = 0;
    elseif (a2 > ul(2) - 1e-8)
        a2 = ul(2);
    end
    
    %if the change in the first lagrange multipler is small return zero
    if (abs(a2-lambda2) < eps*(a2+lambda2+eps))
        return
    end
    a1 = lambda1 + s * (lambda2-a2);
    
    % evaluate threshold or b AND updating the lagrange multipliers and
    % caches errors
    % Although we didn't mention how to update b in Professor's slide, I
    % just follow the 'Original Work of Microsoft', which is performed by Platt's.
    b1 = E1 + Y(i1) * (a1-lambda1) * k11 + Y(i2) * (a2-lambda2) * k12 + b;
    % b is b_old
    b2 = E2 + Y(i1) * (a1-lambda1) * k12 + Y(i2) * (a2-lambda2) * k22 + b;
    b_old = b;
    b = (b1+b2)/2;
    % When both new Lagrange multipliers are at bound and if L is not equal
    % to H, then the interval between b1 and b2 are all thresholds that are
    % consistent with the KKT conditions. In this case, SMO chooses the
    % threshold to be halfway in between b1 and b2.
   
    % Whenever a joint optimization occurs, the stored errors for all
    % non-bound multipliers alpha_k that are not involved in the
    % optimization are updated according to:
    Error = Error + Y(i1) * (a1-lambda1).* K(i1,:) + Y(i2)*(a2-lambda2).*K(i2,:) + b_old - b;
    % update error cache
    Error(i1) = 0;
    Error(i2) = 0;
    lambda(i1) = a1;
    lambda(i2) = a2;
    % Update new lambda
    out = 1;
end