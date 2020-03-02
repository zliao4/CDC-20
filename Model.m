classdef Model
    properties
        fmeanx;
        fmeany;
        fcovx;
        fcovy;
    end
    methods
        function this = Model(data_tr,meas_tr,data_ts,kernel,hyper_param_x,hyper_param_y,noise)
            [this.fmeanx,this.fcovx,~,~,~,~] = gpInf(data_tr,meas_tr(:, 1),data_ts,kernel,hyper_param_x,noise);
            [this.fmeany,this.fcovy,~,~,~,~] = gpInf(data_tr,meas_tr(:, 2),data_ts,kernel,hyper_param_y,noise);
        end
        function this = converge(this, m2)
            [this.fmeanx, this.fcovx] = this.conv(this.fmeanx, m2.fmeanx, this.fcovx, m2.fcovx);
            [this.fmeany, this.fcovy] = this.conv(this.fmeany, m2.fmeany, this.fcovy, m2.fcovy);
        end
        function [mean, cov] = conv(this, mean1, mean2, cov1, cov2)
            sca = 10^-5;
            if (cond(cov1) > 10^3)
                cov1 = cov1 + sca*eye(size(cov1,1));
            end 
            if (cond(cov2) > 10^3)
                cov2 = cov2 + sca*eye(size(cov2,1));
            end 
            function loss = loss_func(alpha)
                loss = 1/2*(logdet(alpha*cov1+(1-alpha)*cov2)-logdet(cov1)*alpha-logdet(cov2)*(1-alpha)) + alpha*(1-alpha)/2*(mean1-mean2)'*(alpha*cov1+(1-alpha)*cov2)*(mean1-mean2);
                loss = -loss;
            end
            func = @loss_func;
            alpha0 = 0;
            [alpha,~,~,~,~,~] = fmincon(func, alpha0, 0, 0, 0, 0, 0, 1);
            cov_tmp = alpha * cov1^-1 + (1-alpha) * cov2^-1;
            mean = cov_tmp^-1 * (alpha * cov1^-1 * mean1 + (1-alpha) * cov2^-1 * mean2);
            cov =  cov_tmp^-1;
        end
        function this = pass_model(this, m2)
            m2.fmeanx = this.fmeanx;
            m2.fmeany = this.fmeany;
            m2.fcovx = this.fcovx;
            m2.fcovy = this.fcovy;
        end
    end
end