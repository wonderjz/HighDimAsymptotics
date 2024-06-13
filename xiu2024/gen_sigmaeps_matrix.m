function matrix = gen_sigmaeps_matrix(n,lowerbd,upperbd)
    %  a diagonal matrix with i.i.d. entries sampled from the uniform distribution U(0.5, 1.5).
    matrix = zeros(n, n);
    for i = 1:n
        matrix(i,i)= unifrnd(lowerbd,upperbd) ;
    end
end