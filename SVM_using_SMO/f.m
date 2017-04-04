function out=f(Y,lambda,b,id,Kernel)

    out = sum(Y'.*lambda.*Kernel(id,:)) - b;
    
end