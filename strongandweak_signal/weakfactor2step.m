% myCluster = parcluster('local');
% myCluster.NumWorkers = 10;
% parp = parpool(10, 'IdleTimeout', 30); % idletimeout is the minute


clear;
constlist = [0.1,0.2,0.4,0.5,0.75,1,1.25,1.5,1.75,2,2.25,2.5,2.75,3,3.25,3.5,3.75,4,4.5,5];
MSEridgeendlist = zeros(1,length(constlist));
MSElassoendlist = zeros(1,length(constlist));
MSE2stependlist = zeros(1,length(constlist));
MSEols2stependlist = zeros(1,length(constlist));
MSEoracleolsendlist = zeros(1,length(constlist));


for con= 1:length(constlist)

const = constlist(1,con);
N = 250;
p1 = 3;
p2 = floor(const*N);
p3 = N/2;

T = N;
round = 50; % repeated times

beta1 = ones(p1,1);
beta2 = ones(p2,1)/sqrt(p2);
%beta2 = normrnd(0,1,p2,1);beta2=beta2/norm(beta2);

%beta2 = normrnd(0,1,p2,1)*1.678/(log(p2)^(2)*sqrt(p2)); % 1.58 = (250)^(1/12)
%beta3 = zeros(p3,1);
%beta3 = ones(p3,1)*(1e-20);
SNR = 1/4; % for the weak and noise ratio
beta4 = sqrt(1/SNR); % SNR = 2 = norm(beta2)^2/(beta4^2) i.e. beta4^2 = norm(beta2)^2/SNR;
% beta = cat(1,beta1,beta2,beta3,beta4);
% K = p1+p2+p3+1;
beta = cat(1,beta1,beta2,beta4);
K = p1+p2+1;


MSElistpurelasso = zeros(1,round);
MSElistpureridge = zeros(1,round);
%MSElistenet = zeros(1,round);
MSElistols2step = zeros(1,round);
MSElist2step = zeros(1,round);
MSElistoracleols = zeros(1,round);

Lambda0 = 0;
Lambda1 = logspace(-10,-1,10);
Lambda2 = linspace(0.02,0.1,10);
Lambda3 = linspace(0.2,1,10);
Lambda4 = linspace(2,10,10);
Lambdalist = cat(2,Lambda0,Lambda1,Lambda2,Lambda3,Lambda4);


for i = 1:round

    % T observations for in sample
    X1 = randn(T,p1);
    X2 = randn(T,p2);
    % X3 = randn(T,p3);
    err = randn(T,1);
    % X = cat(2,X1,X2,X3,err);
    % y = (X1*beta1+ X2*beta2 + X3*beta3+ err*beta4);
    X = cat(2,X1,X2);
    y = (X1*beta1+ X2*beta2+err*beta4);
    
    % 1 observation for out of sample
    X1out = randn(1,p1);
    X2out = randn(1,p2);
    % X3out = randn(1,p3);
    errout = randn(1,1);
    % Xout = cat(2,X1out,X2out,X3out,errout);
    % yout = (X1out*beta1+ X2out*beta2 + X3out*beta3+ errout*beta4);
    Xout = cat(2,X1out,X2out);
    yout = (X1out*beta1+ X2out*beta2 + errout*beta4);
    
    % step 1 LASSO with CV 
    %[B1,FitInfo1] = lasso(X,y,'CV',10);
    [B1,FitInfo1] = lasso(X,y,'Standardize',false,'Intercept',false,'CV',10);
    idxLambda1SE1 = FitInfo1.Index1SE;
    coef1 = B1(:,idxLambda1SE1);
    %coefinter1 = FitInfo1.Intercept(idxLambda1SE1);
    yhat = X*coef1;%+coefinter1;
    res = y - yhat; % for step 2
    idxstep1 = find(coef1)';

    % step 2 Ridge with CV
    idxstep2 = find(coef1==0)';

    if size(idxstep2,2)>0
        % Xstep2 = X(:,idxstep2);
        % Lambdalist2step = Lambdalist*10;
        % CVMdl1 = fitrlinear(Xstep2',res,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist2step,'Learner','leastsquares','Regularization','ridge');
        % cvmseresult1 = kfoldLoss(CVMdl1);
        % [cvmsevalue1,cvlambdaind1] = min(cvmseresult1);
        % optlambdastep2 = 2*Lambdalist(cvlambdaind1); % fitrlinear function  
        % coef2 =  (Xstep2'*Xstep2 + optlambdastep2* eye(length(idxstep2)))^(-1)*(Xstep2'*res);


        Mmatrix = eye(size(X(:,idxstep1),1)) - X(:,idxstep1)*( X(:,idxstep1)'*X(:,idxstep1) )^(-1)*X(:,idxstep1)';
        resXstep2aflasso = Mmatrix*X(:,idxstep2);
        resystep2aflasso = Mmatrix*y;
        Lambdalist2step = Lambdalist*10;
        CVMdl1 = fitrlinear(resXstep2aflasso',resystep2aflasso,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist2step,'Learner','leastsquares','Regularization','ridge');
        cvmseresult1 = kfoldLoss(CVMdl1);
        [cvmsevalue1,cvlambdaind1] = min(cvmseresult1);
        optlambdastep2 = 2*Lambdalist(cvlambdaind1); % fitrlinear function 
        coef2 =  (resXstep2aflasso'*resXstep2aflasso + optlambdastep2* eye(size(resXstep2aflasso,2)))^(-1)*(resXstep2aflasso'*resystep2aflasso);
        %projection coef1
        coef1hat = (X(:,idxstep1)'*X(:,idxstep1))^(-1)*X(:,idxstep1)'*(y-X(:,idxstep2)*coef2);

    else 
        coef2 = 0;
        coef1hat = coef1;
    end
    % pure ridge
    CVMdl2 = fitrlinear(X',y,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist,'Learner','leastsquares','Regularization','ridge');
    cvmseresult2 = kfoldLoss(CVMdl2);
    [cvmsevalue2,cvlambdaind2] = min(cvmseresult2);
    optlambdaridge = 2*Lambdalist(cvlambdaind2);
    coefridge =  (X'*X + optlambdaridge* eye(size(X,2)))^(-1)*(X'*y);
    
    % (Oracle)OLS + regularization
    idxforols = [1:3]'; 
    idxolsstep2 = [4:size(X,2)]';
    coefols = ridgeless(y,X(:,idxforols));
    res3 = y- X(:,idxforols)*coefols;

    % if size(idxolsstep2,1)>0
    %     CVMdl3 = fitrlinear(X(:,idxolsstep2)',res3,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist,'Learner','leastsquares','Regularization','ridge');
    %     cvmseresult3 = kfoldLoss(CVMdl3);
    %     [cvmsevalue3,cvlambdaind3] = min(cvmseresult3);
    %     optlambdaridge3 = Lambdalist(cvlambdaind3);
    %     coefols2step =  (X(:,idxolsstep2)'*X(:,idxolsstep2) + optlambdaridge3* eye(size(X(:,idxolsstep2),2)))^(-1)*(X(:,idxolsstep2)'*res3);
    % else
    %     coefols2step = 0;
    % end

    %(Oracle OLS + L2)
    if size(idxolsstep2,1)>0
        Mmatrix = eye(size(X(:,idxforols),1)) - X(:,idxforols)*( X(:,idxforols)'*X(:,idxforols) )^(-1)*X(:,idxforols)';
        resXstep2 = Mmatrix*X(:,idxolsstep2);
        resystep2 = Mmatrix*y;
        CVMdl3 = fitrlinear(resXstep2',resystep2,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist,'Learner','leastsquares','Regularization','ridge');
        cvmseresult3 = kfoldLoss(CVMdl3);
        [cvmsevalue3,cvlambdaind3] = min(cvmseresult3);
        optlambdaridge3 = 2*Lambdalist(cvlambdaind3); % Ridge penality in fitrlinear is lambda/2
        coefols2step =  (resXstep2'*resXstep2 + optlambdaridge3* eye(size(resXstep2,2)))^(-1)*(resXstep2'*resystep2);
        coefolshat = (X(:,idxforols)'*X(:,idxforols))^(-1)*X(:,idxforols)'*(y-X(:,idxolsstep2)*coefols2step);
        
        % test the two kinds of Cross-Validation
        % CVMdl4 = fitrlinear(X(:,idxolsstep2)',y-X(:,idxforols)*coefolshat,'ObservationsIn','columns','KFold',10,'Lambda',Lambdalist,'Learner','leastsquares','Regularization','ridge');
        % cvmseresult4 = kfoldLoss(CVMdl4);
        % [cvmsevalue4,cvlambdaind4] = min(cvmseresult4);        
        % optlambdaridge4 = 2*Lambdalist(cvlambdaind4); % Ridge penality in fitrlinear is lambda/2 
        % coefols2steptry =  (X(:,idxolsstep2)'*X(:,idxolsstep2) + optlambdaridge4* eye(size(X(:,idxolsstep2),2)))^(-1)*(X(:,idxolsstep2)'*(y-X(:,idxforols)*coefolshat));
        % coefols2step_diff = coefols2step-coefols2steptry;
        % rescoefols2step_diff = (coefols2step_diff'*coefols2step_diff)/(coefols2step'*coefols2step);
        % %disp(['CVbetahat_diff: ',num2str(coefols2step),'doubleCVbetahat: ',num2str(coefols2steptry)]); % compare double CV and single CV
        % disp(['CVbetahat_diff: ',num2str(rescoefols2step_diff)]);
    else
        coefols2step = 0;
        coefolshat = coefols;
    end
    % my feasible double ridge with thereshold

    % idxforLmatrix1 = find(abs(coefridge)>0.5);
    % idxforLmatrix2 = find(abs(coefridge)>0.05 & abs(coefridge)<=0.5 );
    % idxforLmatrix3 = find(abs(coefridge)>0.005 & abs(coefridge)<=0.05);
    % idxforLmatrix4 = find(abs(coefridge)<=0.005);
    % diaglist = zeros(1,length(coefridge));
    % diaglist(idxforLmatrix4) = 0.97;
    % diaglist(idxforLmatrix3) = 0.98;
    % diaglist(idxforLmatrix2) = 0.99;
    % diaglist(idxforLmatrix1) = 1;
    % Lmatrix = diag(diaglist);
    % 
    % errorlist = zeros(1,length(Lambdalist));
    % for l =1:length(Lambdalist)
    %     mylambda = Lambdalist(l);
    %     errork = 0;
    %     for k = 1:10
    %         groupsize = floor(size(X,1)/10);
    %         idxleft = [k:(k+groupsize-1)];
    %         idxuse = setdiff(1:size(X,1),idxleft);
    %         Xtrain = X(idxuse,:);
    %         ytrain = y(idxuse,:);
    %         %Lmatrixuse = Lmatrix(idxuse,idxuse);
    %         coefcrossols2step =  (Xtrain'*Xtrain + mylambda* Lmatrix)^(-1)*(Xtrain'*ytrain);
    %         Xtest = X(idxleft,:);
    %         ytest = y(idxleft,:);
    %         crosserror = ytest - Xtest*coefcrossols2step;
    %         errork = errork+ (crosserror'*crosserror)/length(crosserror);
    %     end
    %     errorlist(k) = errork;
    % end
    % [Minols2steperror,idxformylambda] = min(errorlist);
    % myoptlambda = Lambdalist(idxformylambda);
    % coefols2step =  (X'*X + myoptlambda* Lmatrix)^(-1)*(X'*y);


    %% prediction

    % step 1 (also pure lasso)
    youthatstep1 = Xout*coef1; %+coefinter1;
    resoutlasso = yout - youthatstep1;
    MSEpurelasso = resoutlasso^2; % mse
    %MSEpurelasso = norm(resoutlasso,"fro")/T; % mse
    MSElistpurelasso(1,i) = MSEpurelasso;

    % step 2
    Xoutstep2 = Xout(:,idxstep2);
    if coef2 == 0
        youthatstep2 = 0;
    else       
        youthatstep2 =  Xoutstep2*coef2;
    end
    %youthatstep2 = coef2(1) + Xoutstep2*coef2(2:end);
    resfinal = yout - Xout(:,idxstep1)*coef1hat - youthatstep2;
    MSE2step = resfinal^2; % mse
    %MSE2step = norm(resfinal,"fro")/T; % mse
    MSElist2step(1,i) = MSE2step;
    

    %ridge
    %youthatridge = coefridge(1) + Xout*coefridge(2:end);
    youthatridge = Xout*coefridge;
    resridge = yout - youthatridge;
    MSEridge = resridge^2; % mse
    %MSEridge = norm(resridge,"fro")/T; % mse
    MSElistpureridge(1,i) = MSEridge;

    % enet
    %[B2,FitInfo2] = lasso(Xstep2,res,'Alpha',0.5,'CV',10,'Intercept',false); % alpha for L1 so this is ridge
    
    % only Oracle OLS
    resoracleols = yout - Xout(:,idxforols)*coefols;
    MSEoracleols = resoracleols^2; % mse
    MSElistoracleols(1,i) = MSEoracleols; 

    %(Oracle)OLS + Ridge
    res3final = yout - Xout(:,idxforols)*coefolshat - Xout(:,idxolsstep2)*coefols2step;
    MSEolsridge = res3final^2; % mse
    %MSEolsridge = norm(res3final,"fro")/T; % mse
    MSElistols2step(1,i) = MSEolsridge;

    %(Oracle)OLS + Lasso
    % res3final = yout - Xout(:,idxforols)*coefols - Xout(:,idxolsstep2)*coefols2step;
    % MSEolsridge = res3final^2; % mse
    % MSElistols2step(1,i) = MSEolsridge;


    % res3final = yout - Xout*coefols2step;
    % MSEols2step = norm(res3final,"fro")/T; % mse
    % MSElistols2step(1,i) = MSEols2step;
    

end

MSEridgeend = mean(MSElistpureridge);
MSE2stepend = mean(MSElist2step);
MSElassoend = mean(MSElistpurelasso);
MSEols2stepend = mean(MSElistols2step);
MSEoracleolsend = mean(MSElistoracleols);
dispm = ['const: ',num2str(constlist(con)),' MSE_ridge: ' ,num2str(MSEridgeend), ' MSE_2step: ' ,num2str(MSE2stepend),' MSE_lasso: ' ,num2str(MSElassoend),' MSE_ols2step: ' ,num2str(MSEols2stepend),' MSE_OOLS: ' ,num2str(MSEoracleolsend)];
disp(dispm); 

MSEridgeendlist(1,con) = MSEridgeend;
MSE2stependlist(1,con) = MSE2stepend;
MSElassoendlist(1,con) = MSElassoend;
MSEols2stependlist(1,con) = MSEols2stepend;
MSEoracleolsendlist(1,con) = MSEoracleolsend;
end


scatter(constlist,MSElassoendlist,"filled",'b'); % blue (LASSO)
hold on   
scatter(constlist,MSE2stependlist,"filled",'k'); % black (2step:LASSO+Ridge)
hold on 
scatter(constlist,MSEridgeendlist,"filled",'r');% red (Ridge)
hold on
scatter(constlist,MSEols2stependlist,"filled",'g');% green (OracleOLS+Ridge)
hold on
scatter(constlist,MSEoracleolsendlist,"filled",'m'); % purple magenta" (OracleOLS)
ylim([0 2.5]);

