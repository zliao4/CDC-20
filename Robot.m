classdef Robot
    properties
        idx; % index of the robot
        cur_t; % current time
        
        % motion  
        state; % = [x, y, tehta, v],  the state of the robot
        ctr; %  = [w, a], the control input of the robot
        
        prop_targets; % property of targets
        
        num_target = []; % number of targets
        
        grad = [];
        next_poses = []; % the robots' position in next time frame
        
        entropy;
    end
    %% 
    methods
        function this = Robot(idx, pos, ctr, num_target)
            if nargin<4
              num_target = 1;
            end
            this.idx = idx;
            this.state = pos;
            this.ctr = ctr;
            this.cur_t = 0;
            this.num_target = num_target;
            for i = 1:num_target
                this.prop_targets = [this.prop_targets, TargetProperty];
            end
        end
        
        %% Measure the optical flow model of this robot
        function this = measure(this, obj)
            m_x = obj.x + normrnd(0, 0.1, size(obj.x));
            m_y = obj.y + normrnd(0, 0.1, size(obj.y));
            
            % distance between key points of the robot and target
            d = norm([this.state(1) - mean(m_x), this.state(2) - mean(m_y)]); 
            
            % set whether the robot can oberserve the object
            if d >= 0 && d <= 5
                m_fx = obj.fx + normrnd(0, 0.1, size(obj.fx));
                m_fy = obj.fy + normrnd(0, 0.1, size(obj.fy));
                m_t = obj.t;
                this.prop_targets(obj.idx).num_keypoints = obj.s;
                this.prop_targets(obj.idx).obs = [this.prop_targets(obj.idx).obs; m_x, m_y, m_t * ones(size(m_x)), m_fx, m_fy, (1:obj.s)'];
                this.prop_targets(obj.idx).new_obs = [this.prop_targets(obj.idx).new_obs; m_x, m_y, m_t * ones(size(m_x)), m_fx, m_fy, (1:obj.s)'];
            end
            
        end
        
        %% the robot move for one step
        function this = move(this, dt)
            this.state = this.trans(this.state, this.ctr, dt);
            this.cur_t = this.cur_t + dt;
            this.next_poses = [];
        end
        function state = trans(this, state, ctr, dt)
            x = state(1); y = state(2); theta = state(3); v = state(4);
            w = ctr(1); a = ctr(2);
            x = x + v * cos(theta) * dt;
            y = y + v * sin(theta) * dt;
            theta = theta + w * dt;
            v = v + a * dt;
            state = [x, y, theta, v];
        end
        
        % get dataset form other robots
        function [this, r] = addData(this, r)
            for i = 1:this.num_target
                this.prop_targets(i).obs = [this.prop_targets(i).obs; r.prop_targets(i).new_obs];
                % clean the new obs in r
                r.prop_targets(i).new_obs = [];
                if this.prop_targets(i).num_keypoints == 0
                    this.prop_targets(i).num_keypoints = r.prop_targets(i).num_keypoints;
                end
            end
        end
        
        %% learn GP model from the observation
        function this  = learnGP(this, range)
            % load flow and locations
            for i = 1:this.num_target
                flow_d = this.prop_targets(i).obs;
                if ~isempty(flow_d)
                    flow_d = flow_d(flow_d(:, 3) >= this.cur_t - range, :);
                    
                    if ~isempty(flow_d)
                        % GP regression model
                        thetaX0 = [0.01, 0.1, 0.01, 0.1];
                        predicted_x = fitrgp(flow_d(:,1:3), flow_d(:,4),...
                            'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                            'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc','BasisFunction','none');
                        predicted_y = fitrgp(flow_d(:,1:3), flow_d(:,5),...
                            'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                            'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc','BasisFunction','none');

                        % store the GP model and hyper params
                        this.prop_targets(i).GP_Model = {predicted_x, predicted_y};
                        this.prop_targets(i).hyper_param_x = predicted_x.KernelInformation.KernelParameters;
                        this.prop_targets(i).hyper_param_y = predicted_y.KernelInformation.KernelParameters;
                    end
                end
            end
        end
        
        %% predicted the movement of the object based on the GP model
        function this = pre_prediction(this, future_frame, dt, range)
            for i = 1:this.num_target
                target_property = this.prop_targets(i);
                flow_d = target_property.obs;
                
                % load GP model
                model = this.prop_targets(i).GP_Model;
                
                predicted_x = model{1};
                predicted_y = model{2};

                % store the pre_prediction model
                Data = [];
                for j = 1:target_property.num_keypoints
                    % merge the obervation for the same point by compute the average
                    [idxf,~] = find((flow_d(:,3) == this.cur_t) .*(flow_d(:,6) == j));
                    if numel(idxf) > 1 
                        Data = [Data; mean(flow_d(idxf,:))];
                    else
                        Data = [Data; flow_d(idxf,:)];
                    end
                end
                
                if ~isempty(Data)
                    data_tr = [];
                    meas_tr = [];
                    tmp_obs = target_property.obs;
                    tmp_obs = tmp_obs(tmp_obs(:, 3) >= this.cur_t - range, :);
                    data_tr = [data_tr; tmp_obs(:, 1:3)];
                    meas_tr = [meas_tr; tmp_obs(:, 4:5)];
                    
                    this.prop_targets(i).model = {};
                    this.prop_targets(i).fcov{1} = {};
                    for t = this.cur_t+dt : dt : this.cur_t+dt*future_frame
                        [MeanX, ~] = predict(predicted_x, Data(:,1:3));
                        [MeanY, ~] = predict(predicted_y, Data(:,1:3));
                        Data(:, 1) = Data(:, 1) + MeanX * dt;
                        Data(:, 2) = Data(:, 2) + MeanY * dt;
                        Data(:, 3) = t * ones(size(Data,1),1);
                        this.prop_targets(i).model = [this.prop_targets(i).model, Model(data_tr,meas_tr,Data(:, 1:3),@sptempKernel_2,target_property.hyper_param_x,target_property.hyper_param_y,0.1)];
                        k = int8((t - this.cur_t) / dt);
                        this.prop_targets(i).fcov{1} = [this.prop_targets(i).fcov{1}, {this.prop_targets(i).model(k).fcovx}, {this.prop_targets(i).model(k).fcovy}];
                    end
                end
            end
        end
        
        
        % predicted the movement of the object based on the GP model
        function this = post_prediction(this, future_frame, dt)
            for i = 1:this.num_target
                target_property = this.prop_targets(i);
                flow_d = target_property.obs;

                % start prediction
                Data = [];
                for j = 1:target_property.num_keypoints
                    % merge the obervation for the same point by compute the average
                    [idxf,~] = find((flow_d(:,3) == this.cur_t) .*(flow_d(:,6) == j));
                    if numel(idxf) > 1 
                        Data = [Data; mean(flow_d(idxf,:))];
                    else
                        Data = [Data; flow_d(idxf,:)];
                    end
                end
                if isempty(Data)
                    if ~isempty(target_property.predicted)
                        Data = target_property.predicted;
                        Data = Data(1:target_property.num_keypoints, :);
                    end
                end
                if ~isempty(Data)
                    p = [];
                    for t = this.cur_t+dt : dt : this.cur_t+dt*future_frame
                        k = int8((t - this.cur_t) / dt);
                        Data(:, 1) = Data(:, 1) + target_property.model(k).fmeanx * dt;
                        Data(:, 2) = Data(:, 2) + target_property.model(k).fmeany * dt;
                        Data(:, 3) = t * ones(size(Data,1),1);

                        p = [p; Data(:, 1:3)];
                    end
                    this.prop_targets(i).predicted = p;
                end
            end
        end
        
        % converge the prediction model of two robots
        function [this, r] = converge(this, r, future_frame)
            % converge the prediction
            for i = 1:this.num_target
                if isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                     this.prop_targets(i).model = r.prop_targets(i).model;
                elseif isempty(r.prop_targets(i).model) && ~isempty(this.prop_targets(i).model)
                     r.prop_targets(i).model = this.prop_targets(i).model;
                elseif ~isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                    for t = 1:future_frame
                        this.prop_targets(i).model(t) = this.prop_targets(i).model(t).converge(r.prop_targets(i).model(t));
                    end
                    r.prop_targets(i).model = this.prop_targets(i).model;
                end
            end
            
            % store the converged prediction into fcov
            for i = 1:this.num_target
                if ~isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                    this.prop_targets(i).fcov{1} = {};
                    r.prop_targets(i).fcov{1} = {};
                    for t = 1:future_frame
                        this.prop_targets(i).fcov{1} = [this.prop_targets(i).fcov{1}, {this.prop_targets(i).model(t).fcovx}, {this.prop_targets(i).model(t).fcovy}];
                        r.prop_targets(i).fcov{1} = [r.prop_targets(i).fcov{1}, {r.prop_targets(i).model(t).fcovx}, {r.prop_targets(i).model(t).fcovy}];
                    end
                end
            end
        end
        
        function this = conjugate(this, future_frame)
           sigma = 0.1;
           if ~isempty(this.next_poses)
               for i = 1:this.num_target
                   for t = 1:future_frame
                       if this.next_poses{i}(t) ~= 0
                           n = this.next_poses{i}(t);
                           cov = this.prop_targets(i).fcov{1};
                           if ~isempty(cov)
                               cov1x = cov{2*t-1};
                               cov1y = cov{2*t};
                               mean1x = this.prop_targets(i).model(t).fmeanx;
                               mean1y = this.prop_targets(i).model(t).fmeany;
                               this.prop_targets(i).fcov{1}{2*t-1} = (cov1x^-1 + (n * sigma^2 * eye(size(cov1x)))^-1)^-1;
                               this.prop_targets(i).fcov{1}{2*t} = (cov1y^-1 + (n * sigma^2 * eye(size(cov1y)))^-1)^-1;
                               this.prop_targets(i).model(t).fcovx = this.prop_targets(i).fcov{1}{2*t-1};
                               this.prop_targets(i).model(t).fcovy = this.prop_targets(i).fcov{1}{2*t};
                               this.prop_targets(i).model(t).fmeanx = (cov1x^-1 + (n * sigma^2 * eye(size(cov1x)))^-1)^-1 * cov1x^-1 * mean1x;
                               this.prop_targets(i).model(t).fmeany = (cov1y^-1 + (n * sigma^2 * eye(size(cov1y)))^-1)^-1 * cov1y^-1 * mean1y;
                           end
                       end
                   end
               end
           end
           
           for i = 1:this.num_target
               for t = 1:future_frame
                   cov = this.prop_targets(i).fcov{1};
                   if ~isempty(cov)
                       cov1x = cov{2*t-1};
                       cov1y = cov{2*t};
                       mean1x = this.prop_targets(i).model(t).fmeanx;
                       mean1y = this.prop_targets(i).model(t).fmeany;
                       this.prop_targets(i).fcov{2}{2*t-1} = (cov1x^-1 + (sigma^2 * eye(size(cov1x)))^-1)^-1;
                       this.prop_targets(i).fcov{2}{2*t} = (cov1y^-1 + (sigma^2 * eye(size(cov1y)))^-1)^-1;
                       this.prop_targets(i).model(t).fcovx = this.prop_targets(i).fcov{2}{2*t-1};
                       this.prop_targets(i).model(t).fcovy = this.prop_targets(i).fcov{2}{2*t};
                       this.prop_targets(i).model(t).fmeanx = (cov1x^-1 + (sigma^2 * eye(size(cov1x)))^-1)^-1 * cov1x^-1 * mean1x;
                       this.prop_targets(i).model(t).fmeany = (cov1y^-1 + (sigma^2 * eye(size(cov1y)))^-1)^-1 * cov1y^-1 * mean1y;
                   end
               end
           end
        end
        
        %% plan the control input based on the predicted position of the object
        function [this, r] = planPath(this, r, dt, future_frame)
            function [entropy, loss] = cost(loss, s, ctr, j, entropy)
                gamma = 0.8;
                p = this.prop_targets(j).predicted;
                if ~isempty(p)
                    for i = 1:future_frame
                        s = this.trans(s, ctr(2*i-1:2*i), dt);
                        n=  this.prop_targets(j).num_keypoints;
                        d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                        %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                        w = max(1-(d^2-5*d+6.25)/6.25, 0);

                        fcov = this.prop_targets(j).fcov{2};
                        if ~isempty(fcov)
                            fcovx = fcov{2*i-1}; fcovy = fcov{2*i};
                            
                            fcov_pre = this.prop_targets(j).fcov{1};
                            fcovx_pre = fcov_pre{2*i-1}; fcovy_pre = fcov_pre{2*i};
                            
                            loss = loss + (1-gamma) * gamma^i * 1/2* ((logdet(fcovx)+logdet(fcovy)) - (logdet(fcovx_pre)+logdet(fcovy_pre))) * w; 
                            
                            if d <= 5 && d >= 0
                                entropy = entropy + 1/2*((logdet(fcovx)+logdet(fcovy)));
                            end
                        end   
                    end
                end
            end
            function loss = loss_func(ctr)
                loss = 0;
                entropy = 0;
                s = this.state;
                ctr1 = ctr(1:2*future_frame);
                for j = 1:this.num_target
                    [entropy, loss] = cost(loss, s, ctr1, j, entropy);
                end 
                this.entropy = entropy;
                
                sr = r.state;
                entropy = 0;
                ctr2 = ctr(2*future_frame + 1:4 * future_frame);
                for j = 1:r.num_target
                    [entropy, loss] = cost(loss, sr, ctr2, j, entropy);
                end
                r.entropy = entropy;
            end
            
            A = [zeros([2*future_frame,2*future_frame]); eye(2*future_frame); -eye(2*future_frame)];
            b = [ones([future_frame, 1]) * (3 - this.state(4)); ones([future_frame, 1]) * this.state(4)];
            b2 = [ones([future_frame, 1]) * (3 - r.state(4)); ones([future_frame, 1]) * r.state(4)];
            for j = 1:future_frame
                for k = 1:j
                    A(j, 2 * k) = dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
            end
            for j = future_frame+1:2*future_frame
                for k = 1:j-future_frame
                    A(j, 2 * k) = -dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
            end
                  
            rng default % For reproducibility
            
            func = @loss_func;
            x0 = [repmat(this.ctr, 1, future_frame)'; repmat(r.ctr, 1, future_frame)'];
            opts = optimoptions(@fmincon,'Algorithm','sqp');
            problem = createOptimProblem('fmincon','objective',...
                func,'x0',x0,'Aineq',[A, zeros(size(A)); zeros(size(A)), A],...
                'bineq',[b; b2],'options',opts);
            ms = MultiStart;
            [opt_ctr,~] = run(ms,problem,2);
            this.ctr = opt_ctr(1:2)';
            r.ctr = opt_ctr(2*future_frame+1:2*future_frame+2)';
            opt_ctr2 = opt_ctr(2*future_frame+1:4*future_frame);
            opt_ctr = opt_ctr(1:2*future_frame);
            
            % decide whether next time frame the robot can measure the target
            this.next_poses = cell(1, this.num_target);
            for j = 1:this.num_target
                this.next_poses{j} = zeros(1, future_frame);
                n=  this.prop_targets(j).num_keypoints;
                p = this.prop_targets(j).predicted;
                s = this.state;
                if ~isempty(p)
                    for i = 1:future_frame
                        s = this.trans(s, opt_ctr(2*i-1:2*i), dt);
                        d1 = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); 
                        if d1 >= 0 && d1 <= 5
                            this.next_poses{j}(i) = this.next_poses{j}(i) + 1;
                        end
                    end
                end
                pr = this.prop_targets(j).predicted;
                sr = r.state;
                if ~isempty(pr)
                    for i = 1:future_frame
                        sr = this.trans(sr, opt_ctr2(2*i-1:2*i), dt);
                        d2 = norm([sr(1) - mean(pr(i*n-n+1:i*n, 1)), sr(2) - mean(pr(i*n-n+1:i*n, 2))]);
                        if d2 >= 0 && d2 <= 5
                            this.next_poses{j}(i) = this.next_poses{j}(i) + 1;
                        end
                    end
                end
            end
        end
        
        function [this, r] = pass_next_pos(this, r)
            r.next_poses = this.next_poses;
            this.next_poses = [];
        end
        
        function [this, r] = planPath_random(this, r, dt, future_frame)
            this.ctr = [];
            r.ctr = [];
            for k = 1: future_frame
                this.ctr = [this.ctr, pi/6 * (rand - 0.5) / 5, 5 * (rand - 0.5) / 5];
                r.ctr = [r.ctr, pi/6 * (rand-0.5) / 5, 5 * (rand - 0.5) / 5];
            end
            tmps = this.trans(this.state, this.ctr(1:2), dt);
            tmpsr = this.trans(r.state, r.ctr(1:2), dt);
            for k = 1:2
                if tmps(k) >= 25
                    this.state(k) = this.state(k) - 30;
                elseif tmps(k) <= -5
                    this.state(k) = this.state(k) + 30;
                end
                if tmpsr(k) >= 25
                    r.state(k) = r.state(k) - 30;
                elseif tmpsr(k) <= -5
                    r.state(k) = r.state(k) + 30;
                end
            end
            
            entropy = 0;
            s = this.state;
            sr = r.state;
            for j = 1:this.num_target
                p = this.prop_targets(j).predicted;
                if ~isempty(p)
                    for i = 1:future_frame
                        s = this.trans(s, this.ctr(2*i-1:2*i), dt);
                        n=  this.prop_targets(j).num_keypoints;
                        d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                        fcovx = this.prop_targets(j).model(i).fcovx;
                        fcovy = this.prop_targets(j).model(i).fcovy;
                        if ~isempty(fcovx) && ~isempty(fcovy) && d <= 5 && d >= 0
                           entropy = entropy + 1/2*((logdet(fcovx)+logdet(fcovy)));
                        end
                        
                        sr = this.trans(sr, r.ctr(2*i-1:2*i), dt);
                        d = norm([sr(1) - mean(p(i*n-n+1:i*n, 1)), sr(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                        fcovx = this.prop_targets(j).model(i).fcovx;
                        fcovy = this.prop_targets(j).model(i).fcovy;
                        if ~isempty(fcovx) && ~isempty(fcovy) && d <= 5 && d >= 0
                           entropy = entropy + 1/2*((logdet(fcovx)+logdet(fcovy)));
                        end
                    end   
                end
            end
            this.entropy = entropy;
            r.entropy = entropy;
            this.ctr = this.ctr(1:2);
            r.ctr = r.ctr(1:2);
        end
        
        function [this, r] = planPath_pursue_nearest(this, r, dt, future_frame)
            function [entropy, loss] = cost(loss, s, ctr, j, entropy)
                gamma = 0.8; 
                p = this.prop_targets(j).predicted;
                if ~isempty(p)
                    for i = 1:future_frame
                        s = this.trans(s, ctr(2*i-1:2*i), dt);
                        n=  this.prop_targets(j).num_keypoints;
                        d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                        loss = loss + (1-gamma) * gamma^i * d;

                        fcovx = this.prop_targets(j).model(i).fcovx;
                        fcovy = this.prop_targets(j).model(i).fcovy;
                        if ~isempty(fcovx) && ~isempty(fcovy) && d <= 5 && d >= 0
                           entropy = entropy + 1/2*((logdet(fcovx)+logdet(fcovy)));
                        end
                    end   
                end
            end
            function loss = loss_func(ctr)
                loss = 0;
                entropy = 0;
                s = this.state;
                d = Inf;
                nearest = 0;
                for j = 1:this.num_target
                    p = this.prop_targets(j).predicted;
                    if ~isempty(p)
                        n =  this.prop_targets(j).num_keypoints;
                        tmpd = norm([s(1) - mean(p(1:n, 1)), s(2) - mean(p(1:n, 2))]); % distance of robot and target
                        if tmpd < d
                            d = tmpd;
                            nearest = j;
                        end
                    end
                    [entropy, ~] = cost(loss, s, ctr, j ,entropy);
                end
                if nearest ~= 0
                    [~, loss] =  cost(loss, s, ctr, nearest, entropy);
                end
                this.entropy = entropy;
            end
            function loss = loss_func2(ctr)
                loss = 0;
                entropy = 0;
                s = r.state;
                d = Inf;
                nearest = 0;
                for j = 1:r.num_target
                    p = this.prop_targets(j).predicted;
                    if ~isempty(p)
                        n =  this.prop_targets(j).num_keypoints;
                        tmpd = norm([s(1) - mean(p(1:n, 1)), s(2) - mean(p(1:n, 2))]); % distance of robot and target
                        if tmpd < d
                            d = tmpd;
                            nearest = j;
                        end
                    end
                    [entropy, ~] = cost(loss, s, ctr, j ,entropy);
                end
                if nearest ~= 0
                    [~, loss] =  cost(loss, s, ctr, nearest, entropy);
                end
                r.entropy = entropy;
            end
            
            A = [zeros([2*future_frame,2*future_frame]); eye(2*future_frame); -eye(2*future_frame)];
            b = [ones([future_frame, 1]) * (3 - this.state(4)); ones([future_frame, 1]) * this.state(4)];
            b2 = [ones([future_frame, 1]) * (3 - r.state(4)); ones([future_frame, 1]) * r.state(4)];
            for j = 1:future_frame
                for k = 1:j
                    A(j, 2 * k) = dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
            end
            for j = future_frame+1:2*future_frame
                for k = 1:j-future_frame
                    A(j, 2 * k) = -dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
            end
                  
            rng default % For reproducibility
            
            func = @loss_func;
            x0 = repmat(this.ctr, 1, future_frame)';
            opts = optimoptions(@fmincon,'Algorithm','sqp');
            problem = createOptimProblem('fmincon','objective',...
                func,'x0',x0,'Aineq',A,'bineq',b,'options',opts);
            ms = MultiStart;
            [opt_ctr,~] = run(ms,problem,2);
            this.ctr = opt_ctr(1:2)';
            
            func2 = @loss_func2;
            x0 = repmat(r.ctr, 1, future_frame)';
            opts2 = optimoptions(@fmincon,'Algorithm','sqp');
            problem2 = createOptimProblem('fmincon','objective',...
                func2,'x0',x0,'Aineq',A,'bineq',b2,'options',opts2);
            ms2 = MultiStart;
            [opt_ctr2,~] = run(ms2,problem2,2);
            r.ctr = opt_ctr2(1:2)';
        end
        
        function [this, r1, r2, r3] = planPath_optimal(this, r1, r2, r3, dt, future_frame)
            function [entropy, loss] = cost(loss, s, ctr, j, entropy)
                gamma = 0.8;
                p = [this.prop_targets(j).predicted; r2.prop_targets(j).predicted];
                if ~isempty(p)
                    for i = 1:future_frame
                        s = this.trans(s, ctr(2*i-1:2*i), dt);
                        n=  this.prop_targets(j).num_keypoints;
                        d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                        %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                        w = max(1-(d^2-5*d+6.25)/6.25, 0);

                        fcov = this.prop_targets(j).fcov{2};
                        if ~isempty(fcov)
                            fcovx = fcov{2*i-1}; fcovy = fcov{2*i};
                            
                            fcov_pre = this.prop_targets(j).fcov{1};
                            fcovx_pre = fcov_pre{2*i-1}; fcovy_pre = fcov_pre{2*i};
                            
                            loss = loss + (1-gamma) * gamma^i * 1/2* ((logdet(fcovx)+logdet(fcovy)) - (logdet(fcovx_pre)+logdet(fcovy_pre))) * w; 
                            
                            if d <= 5 && d >= 0
                                entropy = entropy + 1/2*((logdet(fcovx)+logdet(fcovy)));
                            end
                        end   
                    end
                end
            end
            function loss = loss_func(ctr)
                loss = 0;
                entropy = 0;
                s = this.state;
                ctr1 = ctr(1:2*future_frame);
                for j = 1:this.num_target
                    [entropy, loss] = cost(loss, s, ctr1, j, entropy);
                end 
                this.entropy = entropy;
                
                sr1 = r1.state;
                entropy = 0;
                ctr2 = ctr(2*future_frame + 1:4 * future_frame);
                for j = 1:this.num_target
                    [entropy, loss] = cost(loss, sr1, ctr2, j, entropy);
                end
                r1.entropy = entropy;
                
                sr2 = r2.state;
                entropy = 0;
                ctr3 = ctr(4*future_frame + 1:6 * future_frame);
                for j = 1:this.num_target
                    [entropy, loss] = cost(loss, sr2, ctr3, j, entropy);
                end
                r2.entropy = entropy;
                
                sr3 = r3.state;
                entropy = 0;
                ctr4 = ctr(6*future_frame + 1:8 * future_frame);
                for j = 1:this.num_target
                    [entropy, loss] = cost(loss, sr3, ctr4, j, entropy);
                end
                r3.entropy = entropy;
            end
            
            A = [zeros([2*future_frame,2*future_frame]); eye(2*future_frame); -eye(2*future_frame)];
            b = [ones([future_frame, 1]) * (3 - this.state(4)); ones([future_frame, 1]) * this.state(4)];
            b2 = [ones([future_frame, 1]) * (3 - r1.state(4)); ones([future_frame, 1]) * r1.state(4)];
            b3 = [ones([future_frame, 1]) * (3 - r2.state(4)); ones([future_frame, 1]) * r2.state(4)];
            b4 = [ones([future_frame, 1]) * (3 - r3.state(4)); ones([future_frame, 1]) * r3.state(4)];
            for j = 1:future_frame
                for k = 1:j
                    A(j, 2 * k) = dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
                b3 = [b3; pi/6; 5];
                b4 = [b4; pi/6; 5];
            end
            for j = future_frame+1:2*future_frame
                for k = 1:j-future_frame
                    A(j, 2 * k) = -dt;
                end
                b = [b; pi/6; 5];
                b2 = [b2; pi/6; 5];
                b3 = [b3; pi/6; 5];
                b4 = [b4; pi/6; 5];
            end
                  
            rng default % For reproducibility
            
            A_4 = [A, zeros(size(A)), zeros(size(A)), zeros(size(A));
                   zeros(size(A)), A, zeros(size(A)), zeros(size(A));
                   zeros(size(A)), zeros(size(A)), A, zeros(size(A));
                   zeros(size(A)), zeros(size(A)), zeros(size(A)), A];
            func = @loss_func;
            x0 = [repmat(this.ctr, 1, future_frame)'; repmat(r1.ctr, 1, future_frame)';
                  repmat(r2.ctr, 1, future_frame)'; repmat(r3.ctr, 1, future_frame)'];
            opts = optimoptions(@fmincon,'Algorithm','sqp');
            problem = createOptimProblem('fmincon','objective',...
                func,'x0',x0,'Aineq',A_4,...
                'bineq',[b; b2; b3; b4],'options',opts);
            ms = MultiStart;
            [opt_ctr,~] = run(ms,problem,2);
            this.ctr = opt_ctr(1:2)';
            r1.ctr = opt_ctr(2*future_frame+1:2*future_frame+2)';
            r2.ctr = opt_ctr(4*future_frame+1:4*future_frame+2)';
            r3.ctr = opt_ctr(6*future_frame+1:6*future_frame+2)';
        end
    end
end