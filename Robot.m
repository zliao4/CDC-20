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
        model;
        
        % predicted positions
        predicted; 
        
        % measurement
        num_target; % the number of targets
        num_key_points; % the number of key_points for each target
        obs; % measurement of the objects
        new_obs; % newly added measurement
        
        grad = [];
        next_poses = []; % the robots' position in next time frame
    end
    
    methods
        function this = Robot(idx, pos, ctr, num_target)
            if nargin<4
              num_target = 1;
            end
            this.idx = idx;
            this.state = pos;
            this.ctr = ctr;
            this.cur_t = 0;
            this.obs = cell(1, num_target);
            this.new_obs = cell(1, num_target);
            this.num_key_points = cell(1, num_target);
            this.hyper_param_x= cell(1, num_target);
            this.hyper_param_y= cell(1, num_target);
            this.GP_Model= cell(1, num_target);
            this.model = cell(1, num_target);
            for i = 1:num_target
                this.GP_Model{i} = cell(1, 2);
            end
            this.predicted = cell(1, num_target);
            this.num_target = num_target;
        end
        
        % Measure the optical flow model of this robot
        function this = measure(this, obj)
            m_x = obj.x + normrnd(0, 0.1, size(obj.x));
            m_y = obj.y + normrnd(0, 0.1, size(obj.y));
            
            % distance between key points of the robot and target
            d = norm([this.state(1) - mean(m_x), this.state(2) - mean(m_y)]); 
            w = 1-(d^2-5*d+6.25)/6.25;  % observation weight
            %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1;
            
            % set whether the robot can oberserve the object
            if w >= 0
                m_fx = obj.fx + normrnd(0, 0.1, size(obj.fx));
                m_fy = obj.fy + normrnd(0, 0.1, size(obj.fy));
                m_t = obj.t;
                this.num_key_points{obj.idx} = obj.s;
                this.obs{obj.idx} = [this.obs{obj.idx}; m_x, m_y, m_t * ones(size(m_x)), m_fx, m_fy, (1:obj.s)'];
                this.new_obs{obj.idx} = [this.new_obs{obj.idx}; m_x, m_y, m_t * ones(size(m_x)), m_fx, m_fy, (1:obj.s)'];
            end
        end
        
        % the robot move for one step
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
                this.obs{i} = [this.obs{i}; r.new_obs{i}];
                % clean the new obs in r
                r.new_obs = cell(size(r.new_obs));
            end
            this.num_key_points = r.num_key_points;
        end
        
        % learn GP model from the observation
        function this  = learnGP(this)
            % load flow and locations
            for i = 1:size(this.obs, 2)
                flow_d = this.obs{i};
                if ~isempty(flow_d)
                    flow_d = flow_d(flow_d(:, 3) >= this. cur_t - 5, :);
                    
                    if ~isempty(flow_d)
                        % GP regression model
                        thetaX0 = [0.01, 0.1, 0.01, 0.1];
                        predicted_x = fitrgp(flow_d(:,1:3), flow_d(:,4),...
                            'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                            'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc');
                        predicted_y = fitrgp(flow_d(:,1:3), flow_d(:,5),...
                            'KernelFunction',@sptempKernel_2,'KernelParameters',thetaX0,...
                            'FitMethod','sd','PredictMethod','sd','Optimizer','fminunc');

                        % store the GP model and hyper params
                        this.GP_Model{i} = {predicted_x, predicted_y};
                        this.hyper_param_x{i} =  predicted_x.KernelInformation.KernelParameters;
                        this.hyper_param_y{i} = predicted_y.KernelInformation.KernelParameters;
                    end
                end
            end
        end
        
        % predicted the movement of the object based on the GP model
        function this = prediction(this, future_frame, dt)
            for i = 1:this.num_target
                flow_d = this.obs{i};
                
                % load GP model
                model = this.GP_Model{i};
                
                predicted_x = model{1};
                predicted_y = model{2};

                % start prediction
                Data = [];
                for j = 1:this.num_key_points{i}
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
                    for t = this.cur_t + dt:dt:this.cur_t + dt*future_frame
                        [MeanX,~] = predict(predicted_x, Data(:,1:3));
                        [MeanY,~] = predict(predicted_y, Data(:,1:3));
                        Data(:, 1) = Data(:, 1) + MeanX * dt;
                        Data(:, 2) = Data(:, 2) + MeanY * dt;
                        Data(:, 3) = t * ones(size(Data,1),1);
                        p = [p; Data(:, 1:3)];
                    end
                    this.predicted{i} = p;
                    
                    tmp_obs = this.obs{i};
                    data_tr = tmp_obs(:, 1:3);
                    meas_tr = tmp_obs(:, 4:5);
                    data_ts = p;
                    this.model{i} = Model(data_tr,meas_tr,data_ts,@sptempKernel_2,this.hyper_param_x{i},this.hyper_param_y{i},0.1);
                end
            end
        end
        
        % plan the control input based on the predicted position of the object
        function [this, r] = planPath(this, r, dt)
            function loss = loss_func(ctr)
                loss = 0;
                s = this.state;
                for j = 1:this.num_target
                    p = this.predicted{j};
                    if ~isempty(p)
                        for i = 1:2
                            s = this.trans(s, ctr(2*i-1:2*i), dt);
                            n=  this.num_key_points{j};
                            d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                            %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                            w = max(0, 1-(d^2-5*d+6.25)/6.25);

                            % compute variance var
                            data_tr = [];
                            measure_tr = [];
                            tmp_obs = this.obs{j};
                            tmp_obs = tmp_obs(tmp_obs(:, 3) >= this. cur_t - 5, :);
                            data_tr = [data_tr; tmp_obs(:, 1:3)];
                            measure_tr = [measure_tr; tmp_obs(:, 4:5)];
                            data_ts = p;
                            if isempty(this.next_poses)
                                [~,fcovx,~,~,~,~] = gpInf(data_tr,measure_tr(:, 1),data_ts((i-1)*this.num_key_points{j}+1:i*this.num_key_points{j},:),@sptempKernel_2,this.hyper_param_x{j},0.1);
                                [~,fcovy,~,~,~,~] = gpInf(data_tr,measure_tr(:, 2),data_ts((i-1)*this.num_key_points{j}+1:i*this.num_key_points{j},:),@sptempKernel_2,this.hyper_param_y{j},0.1);
                            else
                                [~,fcovx,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts((i-1)*this.num_key_points{j}+1:i*this.num_key_points{j},:),@sptempKernel_2,this.hyper_param_x{j},0.1);
                                [~,fcovy,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts((i-1)*this.num_key_points{j}+1:i*this.num_key_points{j},:),@sptempKernel_2,this.hyper_param_y{j},0.1);
                            end
                            sca = 10^-10;
                            if(cond(fcovx) > 10^5)
                                fcovx = fcovx + sca*eye(size(fcovx,1));
                            end
                            if(cond(fcovy) > 10^5)
                                fcovy = fcovy + sca*eye(size(fcovy,1));
                            end

                            loss = loss + 1/2*(logdet(fcovx)+logdet(fcovy)) * w;
                        end
                    end
                end
            end
            function loss = loss_func2(ctr)
                loss = 0;
                s = r.state;
                for j = 1:r.num_target
                    p = this.predicted{j};
                    if ~isempty(p)
                        for i = 1:2
                            s = r.trans(s, ctr(2*i-1:2*i), dt);
                            n=  this.num_key_points{j};
                            d = norm([s(1) - mean(p(i*n-n+1:i*n, 1)), s(2) - mean(p(i*n-n+1:i*n, 2))]); % distance of robot and target
                            %w = exp(5*d-5)./(1+exp(5*d-5)) + exp(20-5*d)./(1+exp(20-5*d))-1; % weight based on distance
                            w = max(0, 1-(d^2-5*d+6.25)/6.25);

                            % compute variance var
                            data_tr = [];
                            measure_tr = [];
                            tmp_obs = this.obs{j};
                            tmp_obs = tmp_obs(tmp_obs(:, 3) >= r. cur_t - 5, :);
                            data_tr = [data_tr; tmp_obs(:, 1:3)];
                            measure_tr = [measure_tr; tmp_obs(:, 4:5)];
                            data_ts = p;
                            if isempty(this.next_poses)
                                [~,fcovx,~,~,~,~] = gpInf(data_tr,measure_tr(:, 1),data_ts((i-1)*n+1:i*n,:),@sptempKernel_2,this.hyper_param_x{j},0.1);
                                [~,fcovy,~,~,~,~] = gpInf(data_tr,measure_tr(:, 2),data_ts((i-1)*n+1:i*n,:),@sptempKernel_2,this.hyper_param_y{j},0.1);
                            else
                                [~,fcovx,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts((i-1)*n+1:i*n,:),@sptempKernel_2,this.hyper_param_x{j},0.1);
                                [~,fcovy,~,~,~,~] = gpInf([data_tr;this.next_poses],[],data_ts((i-1)*n+1:i*n,:),@sptempKernel_2,this.hyper_param_y{j},0.1);
                            end
                            sca = 10^-10;
                            if(cond(fcovx) > 10^5)
                                fcovx = fcovx + sca*eye(size(fcovx,1));
                            end
                            if(cond(fcovy) > 10^5)
                                fcovy = fcovy + sca*eye(size(fcovy,1));
                            end

                            loss = loss + 1/2*(logdet(fcovx)+logdet(fcovy)) * w;
                        end
                    end
                end
            end
            A = [0, dt, 0, 0;
                 0, dt, 0, dt;
                 0, -dt, 0, 0;
                 0, -dt, 0, -dt];
            b = [3 - this.state(4);
                 3 - this.state(4);
                 this.state(4);
                 this.state(4)];
            func = @loss_func;
            x0 = repmat(this.ctr, 1, 2);
            [opt_ctr,~,~,~,~,grad] = fmincon(func, x0, A, b, zeros(1, 4), 0, repmat([-pi/6, -5], 1, 2), repmat([pi/6, 5], 1, 2));
            this.grad = [this.grad;  grad', func(opt_ctr)];
            this.ctr = opt_ctr(1:2);
            
            func2 = @loss_func2;
            x02 = repmat(r.ctr, 1, 2);
            [opt_ctr2,~,~,~,~,grad2] = fmincon(func2, x02, A, b, zeros(1, 4), 0, repmat([-pi/6, -5], 1, 2), repmat([pi/6, 5], 1, 2));
            r.grad = [r.grad;  grad2', func2(opt_ctr2)];
            r.ctr = opt_ctr2(1:2);
            
            % decide whether next time frame the robot can measure the target
            p1 = this.trans(this.state, this.ctr, dt);
            for j = 1:this.num_target
                p = this.predicted{j};
                if ~isempty(p)
                    d1 = norm(mean(p(1:this.num_key_points{j}, 1)) - p1(1), mean(p(1:this.num_key_points{j}), 2) - p1(2));
                    w1 = 1-(d1^2-5*d1+6.25)/6.25;
                    %w1 = exp(5*d1-5)./(1+exp(5*d1-5)) + exp(20-5*d1)./(1+exp(20-5*d1))-1;
                    if w1 >= 0
                        this.next_poses = [this.next_poses; p1(1:2), p(1,3)];
                        break;
                    end
                end
            end
            p2 = r.trans(r.state, r.ctr, dt);
            for j = 1:this.num_target
                pr = this.predicted{j};
                if ~isempty(pr)
                    d2 = norm(mean(pr(1:this.num_key_points{j}, 1)) - p2(1), mean(pr(1:this.num_key_points{j}, 2)) - p2(2));
                    %w2 = exp(5*d2-5)./(1+exp(5*d2-5)) + exp(20-5*d2)./(1+exp(20-5*d2))-1;
                    w2 = 1-(d2^2-5*d2+6.25)/6.25;

                    if w2 >= 0
                        this.next_poses = [this.next_poses; p2(1:2), pr(1,3)];
                        break;
                    end
                end
            end
        end
        
        function [this, r] = pass_next_pos(this, r)
            r.next_poses = this.next_poses;
        end
        
        function this = converge(this, r)
            for i = 1:this.num_target
                this.model{i} = this.model{i}.converge(r.model{i});
            end   
        end
    end
end
 
    
        
    