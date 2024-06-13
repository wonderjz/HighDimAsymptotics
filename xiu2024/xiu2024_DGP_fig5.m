
clear;
% myCluster = parcluster('local');
% myCluster.NumWorkers = 6;
% parp = parpool(6, 'IdleTimeout', 30); % idletimeout is the minute


gamma = 1;

N = 200;
p = gamma*N;


q = 0.5;
sigmabeta = 1;
% tau = 1/(N^(0.3)*4);
tau = 0.001;
lambda = 0.5;
lambda_ridge = lambda/tau;


% simulation, prediction
simunum = 1000;
Deltabetalist = zeros(1,simunum);


for round = 1:simunum
    b0 = zeros(N,1);
    % spike and slab distribution generation
    %b0 = q*0 + (1-q)*normrnd(0, sigmabeta^2/(1-q));  % q*dirac(x)
    for j = 1:N
        randnum = unifrnd(0,1);
        if randnum <q % q = 0.5 here
            b0(j,1) = 0;
        else
            b0(j,1) = normrnd(0,1)* sigmabeta/sqrt(1-q);
        end
    end


    beta0 = sqrt(tau/p) * b0;

    sigma1m = gen_sigma1_matrix(N); % sigma1m fixed in each simulation
    sigma2m = gen_sigma2_matrix(p); % sigma2m fixed in each simulation
    sigmaerr = gen_sigmaeps_matrix(N,0.5,1.5); % (lowerbd=0.5, upperbd=1.5) 

    Z = randn(N,p);
    sigma1m_sqrt = sqrtm(sigma1m);
    sigma2m_sqrt = sqrtm(sigma2m);
    X = sigma1m_sqrt*Z*sigma2m_sqrt;
    error = sqrtm(sigmaerr)*randn(N,1);
    Y = X*beta0 + error;

    % beta_hatridge = (X'*X + (lambda_ridge*p)*eye(p)) \ (X'*Y); 
    % notice!!! in XIU's setting tuning param for ridge should * p

    [B1,FitInfo1] = lasso(X,Y,'Standardize',false,'Intercept',false,'CV',10); % standardize???
    idxLambda1SE1 = FitInfo1.Index1SE;
    beta_hatlasso = B1(:,idxLambda1SE1);

    beta_hat = beta_hatlasso*2*sqrt(N);


    % Zout = randn(N,p);
    % Xout = sigma1m_sqrt*Zout*sigma2m_sqrt;
    % errorout = randn(N,1);
    % Yout = Xout*beta0 + errorout;  
    % Yout_hat = Xout*beta_hat + errorout;


    Deltabetalist(1,round) = (norm(sigma2m_sqrt*(beta_hat-beta0))^2- norm(sigma2m_sqrt*beta0)^2)*p/(N*tau^2);
end


histogram(Deltabetalist,NumBins=100,Normalization="pdf");
xline(0,'--r',LineWidth=1.5);

disp(median(Deltabetalist));
disp(mean(Deltabetalist));






