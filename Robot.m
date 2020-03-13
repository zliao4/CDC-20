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
            %w = 1-(d^2-5*d+6.25)/6.25;  % observation weight
            w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1;
            
            % set whether the robot can oberserve the object
            if w >= 0.1
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
                this.prop_targets(i).num_keypoints = r.prop_targets(i).num_keypoints;
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
                    p = [];
                    
                    data_tr = [];
                    meas_tr = [];
                    tmp_obs = target_property.obs;
                    tmp_obs = tmp_obs(tmp_obs(:, 3) >= this.cur_t - range, :);
                    data_tr = [data_tr; tmp_obs(:, 1:3)];
                    meas_tr = [meas_tr; tmp_obs(:, 4:5)];
                    
                    this.prop_targets(i).model = {};
                    for t = this.cur_t+dt : dt : this.cur_t+dt*future_frame
                        [MeanX, ~] = predict(predicted_x, Data(:,1:3));
                        [MeanY, ~] = predict(predicted_y, Data(:,1:3));
                        Data(:, 1) = Data(:, 1) + MeanX * dt;
                        Data(:, 2) = Data(:, 2) + MeanY * dt;
                        Data(:, 3) = t * ones(size(Data,1),1);
                        this.prop_targets(i).model = [this.prop_targets(i).model, Model(data_tr,meas_tr,Data(:, 1:3),@sptempKernel_2,target_property.hyper_param_x,target_property.hyper_param_y,0.1)];
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
            function model_f = make_model(robot, i)
                pos_f = [-100, -100, 0; -100, -100, 0; -100, -100, 0; -100, -100, 0];
                mea_f = [0, 0; 0, 0; 0, 0; 0, 0];
                tes_f = [0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0];
                model_f = Model(pos_f,mea_f,tes_f,@sptempKernel_2,robot.prop_targets(i).hyper_param_x,robot.prop_targets(i).hyper_param_y,0.1);
            end
            
            % converge the prediction
            for i = 1:this.num_target
                if isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                     for t = 1:future_frame
                        this.prop_targets(i).model = [this.prop_targets(i).model, make_model(r, i)];
                        this.prop_targets(i).model(t) = this.prop_targets(i).model(t).converge(r.prop_targets(i).model(t));
                     end
                elseif isempty(r.prop_targets(i).model) && ~isempty(this.prop_targets(i).model)
                     for t = 1:future_frame
                        r.prop_targets(i).model = [r.prop_targets(i).model, make_model(this, i)];
                        this.model{i}(t) = this.prop_targets(i).model(t).converge(r.prop_targets(i).model(t));
                     end
                elseif ~isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                    for t = 1:future_frame
                        this.prop_targets(i).model(t) = this.prop_targets(i).model(t).converge(r.prop_targets(i).model(t));
                    end
                end
                r.prop_targets(i).model = this.prop_targets(i).model;
            end
            
            % store the converged prediction into fcov
            for i = 1:this.num_target
                if ~isempty(this.prop_targets(i).model) && ~isempty(r.prop_targets(i).model)
                    this.prop_targets(i).fcov{1} = {};
                    r.prop_targets(i).fcov{1} = {};
                    for t = 1:future_frame
                        this.prop_targets(i).fcov{1} = [this.prop_targets(i).fcov{1}, this.prop_targets(i).model(t).fcovx, this.prop_targets(i).model(t).fcovy];
                        r.prop_targets(i).fcov{1} = [r.prop_targets(i).fcov{1}, {r.prop_targets(i).model(t).fcovx}, {r.prop_targets(i).model(t).fcovy}];
                    end
                end
            end
        end
        
        %% plan the control input based on the predicted position of the object
        function [this, r] = planPath(this, r, dt, future_frame)
            function loss = loss_func(ctr)
                loss = 0;
                s = this.state;
                for j = 1:this.num_target
                    p = this.prop_targets(j).predicted;
                    if ~isempty(p)
                        for i = 1:future_frame
                            s = this.trans(s, ctr(2*i-1:2*i), dt);
                            n=  this.prop_targets(j).num_keypoints;
                            d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                            w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                            %w = 1-(d^2-5*d+6.25)/6.25;
                            
                            fcov = this.prop_targets(j).fcov{1};
                            fcovx = fcov{2*i-1}; fcovy = fcov{2*i};
                            if ~isempty(this.prop_targets(j).fcov{2})
                                fcov_pre = this.prop_targets(j).fcov{2};
                                fcovx_pre = fcov_pre{2*i-1}; fcovy_pre = fcov_pre{2*i};
                            else
                                pos_f = [-100, -100, 0; -101, -100, 0; -100, -101, 0; -101, -101, 0];
                                mea_f = [-1; -1; -1; -1];
                                tes_f = [-101, -100, 1; -102, -100, 1; -101, -101, 1; -102, -101, 1];
                                tes_f = [-100, -101, 1; -101, -101, 1; -101, -102, 1; -102, -102, 1];
                                [~,fcovx_pre,~,~,~,~] = gpInf(pos_f, mea_f, tes_f,@sptempKernel_2,this.prop_targets(j).hyper_param_x,0.1);
                                [~,fcovy_pre,~,~,~,~] = gpInf(pos_f, mea_f, tes_f,@sptempKernel_2,this.prop_targets(j).hyper_param_y,0.1);
                                sca = 10^-10;
                                if(cond(fcovx_pre) > 10^5)
                                    fcovx_pre = fcovx_pre + sca*eye(size(fcovx_pre,1));
                                end
                                if(cond(fcovy_pre) > 10^5)
                                    fcovy_pre = fcovy_pre + sca*eye(size(fcovy_pre,1));
                                end
                            end
                            loss = loss + 1/2*((logdet(fcovx)+logdet(fcovy)) - (logdet(fcovx_pre)+logdet(fcovy_pre))) * w;   
                        end
                        alpha = 0.08;
%                         loss = loss + alpha * sum(ctr.^2);
                    end
                end 
            end
            function loss = loss_func2(ctr)
                loss = 0;
                s = r.state;
                for j = 1:r.num_target
                    p = this.prop_targets(j).predicted;
                    if ~isempty(p)
                        for i = 1:future_frame
                            s = r.trans(s, ctr(2*i-1:2*i), dt);
                            n=  this.prop_targets(j).num_keypoints;
                            d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                            w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                            %w = 1-(d^2-5*d+6.25)/6.25;
                            
                            fcov = this.prop_targets(j).fcov{1};
                            fcovx = fcov{2*i-1}; fcovy = fcov{2*i};
                            if ~isempty(this.prop_targets(j).fcov{2})
                                fcov_pre = this.prop_targets(j).fcov{2};
                                fcovx_pre = fcov_pre{2*i-1}; fcovy_pre = fcov_pre{2*i};
                                
                            else
                                pos_f = [-100, -100, 0; -100, -100, 0; -100, -100, 0; -100, -100, 0];
                                mea_f = [0; 0; 0; 0];
                                tes_f = [0, 0, 0; 0, 0, 0; 0, 0, 0; 0, 0, 0];
                                [~,fcovx_pre,~,~,~,~] = gpInf(pos_f, mea_f, tes_f,@sptempKernel_2,this.prop_targets(j).hyper_param_x,0.1);
                                [~,fcovy_pre,~,~,~,~] = gpInf(pos_f, mea_f, tes_f,@sptempKernel_2,this.prop_targets(j).hyper_param_y,0.1);
                                sca = 10^-10;
                                if(cond(fcovx_pre) > 10^5)
                                    fcovx_pre = fcovx_pre + sca*eye(size(fcovx_pre,1));
                                end
                                if(cond(fcovy_pre) > 10^5)
                                    fcovy_pre = fcovy_pre + sca*eye(size(fcovy_pre,1));
                                end
                            end
                            loss = loss + 1/2*((logdet(fcovx)+logdet(fcovy)) - (logdet(fcovx_pre)+logdet(fcovy_pre))) * w;   
                        end
                        alpha = 0.1;
%                         loss = loss + alpha * sum(ctr.^2);
                    end
                end
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
            [opt_ctr,~] = run(ms,problem,20);
            this.ctr = opt_ctr(1:2)';
            
            func2 = @loss_func2;
            x0 = repmat(r.ctr, 1, future_frame)';
            opts2 = optimoptions(@fmincon,'Algorithm','sqp');
            problem2 = createOptimProblem('fmincon','objective',...
                func2,'x0',x0,'Aineq',A,'bineq',b2,'options',opts2);
            ms2 = MultiStart;
            [opt_ctr2,~] = run(ms2,problem2,20);
            r.ctr = opt_ctr2(1:2)';
            
            % decide whether next time frame the robot can measure the target
            p1 = this.trans(this.state, this.ctr, dt);
            for j = 1:this.num_target
                p = this.prop_targets(j).predicted;
                if ~isempty(p)
                    d1 = norm(mean(p(1:this.prop_targets(j).num_keypoints, 1)) - p1(1), mean(p(1:this.prop_targets(j).num_keypoints), 2) - p1(2));
                    %w1 = 1-(d1^2-5*d1+6.25)/6.25;
                    w1 = exp(5*d1-5)./(1+exp(5*d1-5)) + exp(20-5*d1)./(1+exp(20-5*d1))-1;
                    if w1 >= 0.1
                        this.next_poses = [this.next_poses; p1(1:2), p(1,3)];
                        break;
                    end
                end
            end
            p2 = r.trans(r.state, r.ctr, dt);
            for j = 1:this.num_target
                pr = this.prop_targets(j).predicted;
                if ~isempty(pr)
                    d2 = norm(mean(pr(1:this.prop_targets(j).num_keypoints, 1)) - p2(1), mean(pr(1:this.prop_targets(j).num_keypoints, 2)) - p2(2));
                    w2 = exp(5*d2-5)./(1+exp(5*d2-5)) + exp(20-5*d2)./(1+exp(20-5*d2))-1;
                    %w2 = 1-(d2^2-5*d2+6.25)/6.25;
                    if w2 >= 0.1
                        this.next_poses = [this.next_poses; p2(1:2), pr(1,3)];
                        break;
                    end
                end
            end
            
            % update covariance of prediction
            for j = 1:this.num_target
                this.prop_targets(j).fcov(2) = this.prop_targets(j).fcov(3);
                this.prop_targets(j).fcov(3) = cell(1, 1);
            end
        end
        
        function [this, r] = pass_next_pos(this, r)
            r.next_poses = this.next_poses;
        end

    end
end