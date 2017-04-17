function out=learned_function(Y,lambda,b,id,Kernel)

    out = sum(Y'.*lambda.*Kernel(id,:)) - b;
    
end