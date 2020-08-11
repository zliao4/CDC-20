function [p_t, c_t, p, p_r, e] = spline_5_targets_info(A, W, T, num_target, R, num_robot, num_data, dt) 
    p_t = cell(1, num_target);
    c_t = cell(1, num_target);
    p_r = cell(1, num_robot);
    p = cell(1, num_robot);
    e = [];
    future_frame = 5;
    for i = 1:num_robot
        p{i} = cell(1, num_target);
    end

    for i = 1:num_data
        % object move
        for j = 1:num_target
            T(j).a = A{j}(i);
            T(j).w = W{j}(i);
            T(j) = T(j).move(dt);
            p_t{j} = [p_t{j}; T(j).x, T(j).y, T(j).theta * ones(size(T(j).x))];
            c_t{j} = [c_t{j}; mean(T(j).x), mean(T(j).y), T(j).theta];
        end

        for j = 1:num_robot
            R(j) = R(j).move(dt); % robot move
            for k = 1:num_target
                R(j) = R(j).measure(T(k)); % robot measure
            end
            p_r{j} = [p_r{j}; R(j).state, R(j).ctr]; % store position information
        end

        % data passing
        for j = 1:num_robot / 2
            [R(2*j), R(2*j-1)] = R(2*j).addData(R(2*j-1));
        end

        % GP learning
        R(2) = R(2).learnGP(9999);  
        R(4) = R(4).learnGP(9999);

        % make pre_prediction
        R(2) = R(2).pre_prediction(future_frame, dt, 9999); 
        R(4) = R(4).pre_prediction(future_frame, dt, 9999);

        % prediction converge
        [R(4), R(2)] = R(4).converge(R(2), future_frame);
        
        %%%%%% Generate new conjugacy model %%%%%%%%%%%%
        %%%%%% R2 conjugate with sigma^2 * I %%%%%%%%%%%
        %%%%%% Store pre-conju and post-conju %%%%%%%%%%
        R(2) = R(2).conjugate(future_frame);

        R(2) = R(2).post_prediction(future_frame, dt); 
        [R(2), R(1)] = R(2).planPath(R(1), dt, future_frame); % path palnning
        [R(2), R(4)] = pass_next_pos(R(2), R(4)); % data passing
        
        %%%%%% R4 conjugate with R2 passed pose %%%%%%%%
        %%%%%% R4 conjugate with sigma ^2 * I %%%%%%%%%%
        %%%%%% Store pre-conju and post-conju %%%%%%%%%%
        R(4) = R(4).conjugate(future_frame);
        
        R(4) = R(4).post_prediction(future_frame, dt);
        [R(4), R(3)] = R(4).planPath(R(3), dt, future_frame); % path palnning
        R(4).next_poses = [];

        for j = 1:num_robot/2
            for k = 1:num_target
                if ~isempty(R(2*j).prop_targets(k).predicted)
                    p{2*j}{k} = [p{2*j}{k}, {R(2*j).prop_targets(k).predicted}];
                else
                    p{2*j}{k} = [p{2*j}{k}, {[-100, -100]}];
                end
            end
        end
        
        tmpe = 0;
        for j = 1:num_robot
            tmpe = tmpe + R(j).entropy;
        end
        e  =[e, tmpe / 4];
    end
end
