classdef Robot
    properties
        
        % basic info
        idx; % index of the robot
        cur_t; % current time
        
        % motion  
        state; % = [x, y, tehta, v],  the state of the robot
        ctr; %  = [w, a], the control input of the robot
        
        % GP model parameters
        hyper_param_x; % hyper parameters
        hyper_param_y; % hyper parameters
        GP_Model;
        predicted; % predicted positions
        
        % measurement
        obs = []; % measurement of the objects
        new_obs = [];
        
        next_poses = []; % the robots' position in next time frame
    end
    
    methods
        function this = Robot(idx, pos, ctr)
            this.idx = idx;
            this.state = pos;
            this.ctr = ctr;
            this.cur_t = 0;
        end
        
        % Measure the optical flow model of this robot
        function this = measure(this, obj)
            m_x = obj.x + normrnd(0, 0.1);
            m_y = obj.y + normrnd(0, 0.1);
            d = norm([this.state(1) - m_x, this.state(2) - m_y]); % distance of robot and target
            w = max(0, 1-(d^2-5*d+6.25)/6.25);    
            if w > 0.1
                m_fx = obj.fx + normrnd(0, 0.1);
                m_fy = obj.fy + normrnd(0, 0.1);
                m_t = obj.t;
                obs_tmp = Obs(m_x, m_y, m_fx, m_fy, m_t);
                this.obs = [this.obs, obs_tmp];
                this.new_obs = [this.new_obs, obs_tmp];
            end
        end
        
        % the robot move for one step
        function this = move(this, dt)
            this.state = this.trans(this.state, this.ctr, dt);
            this.cur_t = this.cur_t + dt;
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
            this.obs = [this.obs, r.new_obs];
            % clean the new obs in r
            r.new_obs = [];
        end
        
        % learn GP model from the observation
        function this  = learnGP(this)
            % load flow and locations
            flow_d = [];
            for i = 1:size(this.obs, 2)
                tmp_obs = this.obs(i);
                flow_d = [flow_d; tmp_obs.x, tmp_obs.y, tmp_obs.t, tmp_obs.fx, tmp_obs.fy]; 
            end
            
            % GP regression model
            thetaX0 = [0.01, 0.1, 0.01, 0.1];
            predicted_x = fitrgp(flow_d(:,1:3), flow_d(:,4),...
                'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc');
            predicted_y = fitrgp(flow_d(:,1:3), flow_d(:,5),...
                'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc');
            
            this.GP_Model = {predicted_x, predicted_y};
            this.hyper_param_x = predicted_x.KernelInformation.KernelParameters;
            this.hyper_param_y = predicted_y.KernelInformation.KernelParameters;
        end
        
        % predicted the movement of the object based on the GP model
        function this = prediction(this, future_frame, dt)
            % load flow and locations
            flow_d = [];
            for i = 1:size(this.obs, 2)
                tmp_obs = this.obs(i);
                flow_d = [flow_d; tmp_obs.x, tmp_obs.y, tmp_obs.t, tmp_obs.fx, tmp_obs.fy]; 
            end
            
            % load GP model
            predicted_x = this.GP_Model{1};
            predicted_y = this.GP_Model{2};
            
            % start prediction
            [idxf,~] = find(flow_d(:,3) == this.cur_t);
            if numel(idxf) > 1 
                Data = mean(flow_d(idxf,:));
            else
                Data = flow_d(idxf,:);
            end
            p = [];
            for t = this.cur_t + dt:dt:this.cur_t + dt*future_frame
                Data(:, 3) = (t - dt) * ones(size(Data,1),1);
                [MeanX,~] = predict(predicted_x, Data(:,1:3));
                [MeanY,~] = predict(predicted_y, Data(:,1:3));
                Data(:, 1) = Data(:, 1) + MeanX;
                Data(:, 2) = Data(:, 2) + MeanY;
                Data(:, 3) = t * ones(size(Data,1),1);
                p = [p; Data(:, [1:3])];
            end
            this.predicted = p;
        end
        
        % plan the control input based on the predicted position of the object
        function this = planPath(this, dt)
            function loss = loss_func(ctr)
                loss = 0;
                s = this.state;
                p = this.predicted;
                for i = 1:5
                    s = this.trans(s, ctr(2*i-1:2*i), dt);
                    loss = loss + norm([s(1) - p(i, 1), s(2) - p(i, 2)]); 
                end
            end
            function loss = loss_func_2(ctr)
                loss = 0;
                s = this.state;
                p = this.predicted;
                for i = 1:2
                    s = this.trans(s, ctr(2*i-1:2*i), dt);
                    d = norm([s(1) - p(i, 1), s(2) - p(i, 2)]); % distance of robot and target
                    %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                    w = max(0, 1-(d^2-5*d+6.25)/6.25);
                    % compute variance var
                    data_tr = [];
                    measure_tr = [];
                    for i = 1:size(this.obs, 2)
                        tmp_obs = this.obs(i);
                        data_tr = [data_tr; tmp_obs.x, tmp_obs.y, tmp_obs.t];
                        measure_tr = [measure_tr; tmp_obs.fx, tmp_obs.fy];
                    end
                    data_ts = this.predicted;
                    if isempty(this.next_poses)
                        [~,fcovx,~,~,~,~] = gpInf(data_tr,measure_tr,data_ts,@sptempKernel_2,this.hyper_param_x,0.1);
                        [~,fcovy,~,~,~,~] = gpInf(data_tr,measure_tr,data_ts,@sptempKernel_2,this.hyper_param_y,0.1);
                    else
                        [~,fcovx,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts,@sptempKernel_2,this.hyper_param_x,0.1);
                        [~,fcovy,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts,@sptempKernel_2,this.hyper_param_y,0.1);
                    end
                    loss = loss + 1/2*log(det(fcovx) + det(fcovy)) * w;
                end
            end
            A = [0, dt, 0, 0,;
                 0, dt, 0, dt;
                 0, -dt, 0, 0;
                 0, -dt, 0, -dt];
            b = [3 - this.state(4);
                 3 - this.state(4);
                 this.state(4);
                 this.state(4)];
            func = @loss_func_2;
            x0 = repmat(this.ctr, 1, 2);
            opt_ctr = fmincon(func, x0, A, b, zeros(1, 4), 0, repmat([-pi/9, -3], 1, 2), repmat([pi/9, 3], 1, 2));
            this.ctr = opt_ctr(1:2);
        end
        
        % plan the control input of two robots
        function [this, r] = planPath2(this, r, dt)
            this = this.planPath(dt);
            r = r.planPath(dt);
            
            p1 = this.trans(this.state, this.ctr, dt);
            d1 = norm(this.predicted(1, 1) - p1(1), this.predicted(1, 2) - p1(2));
            w1 = max(0, 1-(d1^2-5*d1+6.25)/6.25);
            p2 = r.trans(r.state, r.ctr, dt);
            d2 = norm(r.predicted(1, 1) - p2(1), r.predicted(1, 2) - p2(2));
            w2 = max(0, 1-(d2^2-5*d2+6.25)/6.25);
            
            if w1 > 0.1
                this.next_poses = [this.next_poses; p1(1:2), this.predicted(1,3)];
            end
            if w2 > 0.1
                this.next_poses = [this.next_poses; p2(1:2), this.predicted(1,3)];
            end
        end
        
        function [this, r] = pass_next_pos(this, r)
            r.next_poses = this.next_poses;
        end
    end
end
 
    
        
    