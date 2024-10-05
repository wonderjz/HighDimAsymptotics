function [coefols2step] = RidgeCV(Lambdalist,X,y)
lens = length(Lambdalist);
errorlist = zeros(1,lens);    
for l =1:length(Lambdalist)
    mylambda = Lambdalist(l);
    errork = 0;
    for k = 1:10
        groupsize = floor(size(X,1)/10);
        idxleft = [(k-1)*groupsize+1:(k*groupsize)];
        idxuse = setdiff(1:size(X,1),idxleft);
        Xtrain = X(idxuse,:);
        ytrain = y(idxuse,:);
        %Lmatrixuse = Lmatrix(idxuse,idxuse);
        Lmatrix = eye(size(Xtrain,2));
        coefcrossols2step =  (Xtrain'*Xtrain + mylambda* Lmatrix)^(-1)*(Xtrain'*ytrain);
        Xtest = X(idxleft,:);
        ytest = y(idxleft,:);
        crosserror = ytest - Xtest*coefcrossols2step;
        errork = errork+ (crosserror'*crosserror);
    end
    errorlist(l) = errork;
end

[Minols2steperror,idxformylambda] = min(errorlist);
myoptlambda = Lambdalist(idxformylambda);
coefols2step =  (X'*X + myoptlambda* Lmatrix)^(-1)*(X'*y);

