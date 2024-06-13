function [beta_hat]=ridgeless(y,X)
    if size(X,2)< size(X,1)
        beta_hat = (X'*X)^(-1)*(X'*y);
    else
        [U,S,V] = svd(X,'econ');
        beta_hat = V*S^(-1)*U'*y;
    end