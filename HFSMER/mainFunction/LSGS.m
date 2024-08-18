
function [model_W] = LSGS( X, Y,S, optmParameter,maxm)
   %% optimization parameters
    lambda1           = optmParameter.lambda1;
    lambda2           = optmParameter.lambda2;
    lambda3           = optmParameter.lambda3;
    lambda4           = optmParameter.lambda4;
    gamma            = optmParameter.gamma;
    maxIter          = optmParameter.maxIter;
    miniLossMargin   = optmParameter.minimumLossMargin;
    if isempty(X)
        model_W =zeros(100,maxm);
        return;
    end
   %% initializtion
    num_dim = size(X,2);
    num_labels=size(Y,2);
    XTX = X'*X;
    XTY = X'*Y;

    W   = (XTX + gamma*eye(num_dim)) \ (XTY);
    W_1 = W;
    
    S =S;
    S_1 = S;

    options = [];
    options.Metric = 'Euclidean';
    options.NeighborMode = 'KNN';
    options.k = 10;  % nearest neighbor
    options.WeightMode = 'HeatKernel';
    options.t = 1;
    C = constructW(X,options);
    L = diag(sum(C,2))-C;
    iter    = 1;
    oldloss = 0;
    
    bk   = 1;
    bk_1 = 1; 

   %% proximal gradient
    while iter <= maxIter
       D = diag(1./max(sqrt(sum(W.*W,2)),eps));
        Lip1 = 4*norm(XTX)^2 +4*norm(lambda1*D)^2+4*norm(lambda3*(S))^2  + 4*norm(lambda4*(X'*L*X))^2;
       Lip  = sqrt( Lip1 );
     %% update W
       W_k    = W + (bk_1 - 1)/bk * (W - W_1);
       Gw_s_k = W_k - 1/Lip * ((XTX*W_k  - X'*Y) + lambda1*D*W_k+ lambda3*W_k*S+lambda4*X'*L*X*W_k);
       W_1    = W;
       W      = softthres(Gw_s_k,lambda2/Lip);      
       
       bk_1   = bk;
       bk     = (1 + sqrt(4*bk^2 + 1))/2;
       
       M=X*W-Y;
       predictionLoss   = 0.5*trace(M'*M);
       GlobalLC         = 0.5*trace(W*S*W');
       sparsityW21          = 0.5*L21(W);
       XW=X*W;
       LocalLC          = 0.5*trace(XW'*L*XW);
       sparsityW1        = sum(sum(W~=0));    
       totalloss        = predictionLoss + lambda1*sparsityW21 + lambda2*sparsityW1 + lambda3*GlobalLC+lambda4*LocalLC ;
       loss(iter,1)     = totalloss;
       if abs((oldloss - totalloss)/oldloss) <= miniLossMargin
           break;
       elseif totalloss <=0
           break;
       else
           oldloss = totalloss;
       end
       iter=iter+1;
    end
     model_W = W;
end

%soft thresholding operator
function W = softthres(W_t,lambda,~)
    W = max(W_t-lambda,0) - max(-W_t-lambda,0); 
end

