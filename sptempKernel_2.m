function K = sptempKernel_2(Xm,Xn,hyp)
    % spatio-temporal kernel
    % Chang Liu 3/11/19
    
    % K(X,X') = K_X(x,x')K_T(t,t')    
    % Inputs:
    % Xm is an m-by-d matrix. Xn is an n-by-d matrix. theta is the r-by-1
    % unconstrained parameter vector for kfcn.
    % Outputs:
    % Kmn is an m-by-n matrix of kernel products such that Kmn(i,j) is the
    % kernel product between Xm(i,:) and Xn(j,:). 
    
    
%     switch kernel_type
%         case 'sep' % separable spatio-temporal kernel
            Sm = Xm(:,1:end-1); % spatial component
            Sn = Xn(:,1:end-1); % spatial component
            Tm = Xm(:,end); % temporal component
            Tn = Xn(:,end); % temporal component
            
            % full hyperparams
            % use four parameters. use with squaredExpo
            lls_S = hyp(1); % log of length scale
            lsstd_S = hyp(2); % log of signal std dev
            lls_T = hyp(3);
            lsstd_T = hyp(4);
            
            K_S = squaredExpo(Sm,Sn,lls_S,lsstd_S);
            K_T = squaredExpo(Tm,Tn,lls_T,lsstd_T);
            K = K_S.*K_T;
end

function K = squaredExpo(X,Y,lls,lsstd)
    % squared exponential kernel
    K = (exp(lsstd))^2*exp(-(pdist2(X,Y)).^2/(2*exp(lls)^2));
    % the following wants to add sigma^2I to the kernel. Doesn't seem to
    % work well in practice
%     if size(X,1) == size(Y,1)
%         if (all(abs(X-Y)<1e-3)) % if X == Y
%             K = K+2*eye(size(X,1)); % add term for sensor measurement noise
%         end
%     end
end