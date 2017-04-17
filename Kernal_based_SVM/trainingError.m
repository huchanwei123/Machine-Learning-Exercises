function error = trainingError(lambda, Y, b, K)
    error = 0;
    for i = 1:length(Y)
        out = learned_function(Y,lambda,b,i,K);
        if (out < 0 && Y(i) ~= 0)
            error = error + 1;
        elseif (out > 0 && Y(i) ~= 1)
            error = error + 1;
        end
    end
    % Find out error rate
    error = error/length(Y) * 100;
end