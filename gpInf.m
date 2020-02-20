function [fmean,fcov,alp,Kxx,Kqx,Kxx_n] = gpInf(train_x,train_y,test_x,K,hyp,sig_n)
% input:
% train_x, train_y: training data x (input) and y (measurement)
% test_x: testing data (input)
% K: GP kernel
% hyp: GP hyperparameters: 4D vector
% sig_n: measurement noise: scalar, noise std

% output:
% fmean: mean of GP
% fcov: covariance matrix of GP

% modified on 1/20/20
% added output of Kxx, Kqx, Kxx_n



% GP inference function
% test_x = gpuArray(test_x);
% train_x = gpuArray(train_x);
% hyp = gpuArray(hyp);
% sig_n = gpuArray(sig_n);
% train_y = gpuArray(train_y);

Kqq = K(test_x,test_x,hyp);
Kqx = K(test_x,train_x,hyp);
Kxx = K(train_x,train_x,hyp);
% Qinv = inv(Kxx+sig_n^2*eye(size(train_x,1)));
Kxx_n = Kxx+sig_n^2*eye(size(train_x,1));

% fmean
if (~isempty(train_y))
    %     fmean = Kqx*Qinv*train_y;
    %     fmean = Kqx/Kxx_n*train_y;
    L = chol(Kxx_n);
    alphaHat = (L\(L'\train_y));
    fmean = Kqx*alphaHat;
else
    fmean = [];
end

% alpha
alp = Kqx/Kxx_n;
% fcov = Kqq - Kqx*Qinv*Kqx'; % covariance given training data x
fcov = Kqq - alp*Kqx'; % covariance given training data x

% fmean = gather(fmean);
% fcov = gather(fcov);
% alp = gather(alp);
% Kxx = gather(Kxx);
% Kqx = gather(Kqx);
% Kxx_n = gather(Kxx_n);
end