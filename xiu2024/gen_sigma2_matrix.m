
function matrix = gen_sigma2_matrix(n)
    % apporimated diagonal matrix 

    eigenlist = zeros(1,n);
    for i = 1:n
        eigenlist(1,i)= unifrnd(0.5,1.5) ;
    end

    Q1 = randn(n, n);
    % get orthogonal matrix Q by QR decomposition
    [Q, R] = qr(Q1);

    matrix = zeros(n, n);
    for i =1:n
        eigenval = eigenlist(1,i);
        eigenvec = Q(:,i);
        matrix_temp = eigenval*(eigenvec*eigenvec');
        matrix = matrix + matrix_temp;
    end

end