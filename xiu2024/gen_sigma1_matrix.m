function matrix = gen_sigma1_matrix(n)

    matrix = zeros(n, n);

    for i = 1:n
        for j = 1:n
            powernum = -abs(i-j);           
            matrix(i, j) = 2^powernum;
        end
    end

end