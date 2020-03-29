function [p_t, c_t, p, p_r, e] = optimal_GP(A, W, T, num_target, R, num_robot, num_data, dt) 
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
            R(j) = R(j).move(dt);
            for k = 1:num_target
                R(j) = R(j).measure(T(k)); % robot measure
            end
            p_r{j} = [p_r{j}; R(j).state, R(j).ctr]; % store position information
        end

        % data passing
        for j = 1:num_robot - 1
            [R(4), R(j)] = R(4).addData(R(j));
        end
        
%         % GP learning
%         R(2) = R(2).learnGP(5);  
        R(4) = R(4).learnGP(5);

%         % make pre_prediction
%         R(2) = R(2).pre_prediction(future_frame, dt, 5); 
        R(4) = R(4).pre_prediction(future_frame, dt, 5);
        
%         %%%%%% Generate new conjugacy model %%%%%%%%%%%%
%         %%%%%% R2 conjugate with sigma^2 * I %%%%%%%%%%%
%         %%%%%% Store pre-conju and post-conju %%%%%%%%%%
%         R(2) = R(2).conjugate(future_frame);
% 
%         R(2) = R(2).post_prediction(future_frame, dt); 
%         R(2).next_poses = [];

        
        %%%%%% R4 conjugate with R2 passed pose %%%%%%%%
        %%%%%% R4 conjugate with sigma ^2 * I %%%%%%%%%%
        %%%%%% Store pre-conju and post-conju %%%%%%%%%%
        R(4) = R(4).conjugate(future_frame);
        
        R(4) = R(4).post_prediction(future_frame, dt);
        R(4).next_poses = [];

        for j = 1:num_robot/2
            for k = 1:num_target
                if ~isempty(R(4).prop_targets(k).predicted)
                    p{2*j}{k} = [p{2*j}{k}, {R(4).prop_targets(k).predicted}];
                else
                    p{2*j}{k} = [p{2*j}{k}, {[-100, -100]}];
                end
            end
        end
        
    end
end