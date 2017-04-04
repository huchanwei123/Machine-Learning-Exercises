function error = trainingError(lambda, Y, b, K)
    error = 0;
    for i = 1:length(Y)
        if f(Y,lambda,b,i,K) * Y(i) < 0
            error = error + 1;
        end
    end
    % Find out error rate
    error = error/length(Y) * 100;
end