function [out, lambda, b, Error] = examineExample(i2, Y, lambda, tol, ul, Error, eps, K, b)
    out = 0;
    lambda2 = lambda(i2);

    % check if lambda2 is on-bound-->[YES; NO]=[Evaluate E2 from SVM function; 
    % Take E2 from cache Error vector]
    if ((lambda2 < tol) || (lambda2 > ul(2)-tol))
        E2 = f(Y,lambda,b,i2,K) - Y(i2);
    else
        E2 = Error(i2);
    end
    r2 = E2 * Y(i2);
    
    if ((r2 < -tol) && (lambda2 < ul(2))) || ((r2 > tol) && (lambda2 > 0))
        tmp = find((Error>tol) & (Error<ul(2)-tol));
        if(~isempty(tmp)) && (E2 > 0)
            [~, i1] = max(Error); % B_up
            [check,lambda,Error,b] = takeStep(i1,i2,lambda,Y,b,eps,Error,ul,K,E2,tol);
            if check
                out = 1;
                return;
            end
        elseif (~isempty(tmp)) && (E2 < 0)
            [~, i1] = min(Error); % B_low
            [check,lambda,Error,b] = takeStep(i1,i2,lambda,Y,b,eps,Error,ul,K,E2,tol);
            if check
                out = 1;
                return;
            end            
        end
    % loop over all non-zero and non-C lambda, starting at randmom points
    if (~isempty(tmp))
        startPoint = randi(length(tmp));
    % reorder the tmp matrix
        tmp=[tmp(startPoint:end) tmp(1:startPoint-1)];
            for i1=tmp
                 [check,lambda,Error,b]=takeStep(i1,i2,lambda,Y,b,eps,Error,ul,K,E2,tol);
                if check
                    out = 1;
                    return
                end
            end
    end
    % loop over all lagrange multiplier elements, starting at randmom points
    for i1=1:length(lambda)
        [check,lambda,Error,b] = takeStep(i1,i2,lambda,Y,b,eps,Error,ul,K,E2,tol);
        if check
            out = 1;
            return
        end
    end
end