clc;
clear;
close all;

dbstop if error

%%
% generate the control of the target
x = 0:0.05:20;
y = x.^3 * 139/6600 - x.^2 *1393/2200 + 1729/330*x;
x2 = 20:-0.05:0;
y2 = 15 - 0.7 * x2;
x3 = 8:0.01:12;
y3 = 40 - 4 * x3;
theta = atan2(x.^2*139/2200-x*1393/1100+1729/330, 1);
theta2 = atan2(-0.7 * ones(1, 401), 1) + pi;
theta3 = atan2(-4 * ones(1, 401), 1);
dt = 0.2;
v = [0, sqrt((y(2:end)-y(1:end-1)).^2 + (x(2:end)-x(1:end-1)).^2)/dt];
v2 = [0, sqrt((y2(2:end)-y2(1:end-1)).^2 + (x2(2:end)-x2(1:end-1)).^2)/dt];
v3 = [0, sqrt((y3(2:end)-y3(1:end-1)).^2 + (x3(2:end)-x3(1:end-1)).^2)/dt];
w = [0, (theta(2:end)-theta(1:end-1)) / dt];
w2 = [0, (theta2(2:end)-theta2(1:end-1)) / dt];
w3 = [0, (theta3(2:end)-theta3(1:end-1)) / dt];
w = w(1:end-1);
w2 = w2(1:end-1);
w3 = w3(1:end-1);
a = (v(2:end)-v(1:end-1))/dt;
a2 = (v2(2:end)-v2(1:end-1))/dt;
a3 = (v3(2:end)-v3(1:end-1))/dt;
A = {a, a2, a3};
W = {w, w2, w3};

% generate target object
t = TargetObj(1, [-0.25; -0.25; 0.25; 0.25], [-0.25; 0.25; -0.25; 0.25], theta(1), 0, 0, 0);
t2 = TargetObj(2, [19.75; 19.75; 20.25; 20.25], [0.75; 1.25; 0.75; 1.25], theta2(1), 0, 0, 0);
t3 = TargetObj(3, [7.75; 7.75; 8.25; 8.25], [17.75; 18.25; 17.75; 18.25], theta3(1), 0, 0, 0);
T = [t, t2, t3];
num_target = 3;


R = [];
num_robot = 4;
for j = 1:2
    R = [R, Robot(j, [rand*5+2.5, rand*5+10.5, rand * 2 * pi, 0], [0, 0], num_target)];
end

for j = 3:4
    R = [R, Robot(j, [-rand*5+18.5, -rand*5+9.5, rand * 2 * pi, 0], [0, 0], num_target)];
end

% R = [R, Robot(1, [-rand*5+6, rand*5+19.5, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(2, [rand*5+30, -rand*5+14.25, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(3, [-rand*5+6, rand*5+19.5, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(4, [rand*5+30, -rand*2+14.25, rand*2*pi, 0], [0, 0])];

%%
p_t = cell(1, num_target);
c_t = cell(1, num_target);
p_r = cell(1, num_robot);
colors = ['r', 'g', 'c', 'k', 'y', 'm'];
p = cell(1, num_robot);
for i = 1:num_robot
    p{i} = cell(1, num_target);
end

for i = 1:size(a, 2)
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
    R(2) = R(2).learnGP(5);  
    R(4) = R(4).learnGP(5);
    
    % make pre_prediction
    R(2) = R(2).pre_prediction(3, dt, 5); 
    R(4) = R(4).pre_prediction(3, dt, 5);
    
    % prediction converge
    [R(4), R(2)] = R(4).converge(R(2), 3);
    
    % make post_prediction
    R(2) = R(2).post_prediction(3, dt); 
    R(4) = R(4).post_prediction(3, dt);
    
    
%     for j = 1:2
%         R(2).prop_targets(j).fcov(3) = R(2).prop_targets(j).fcov(1);
%         R(2).prop_targets(j).fcov(1) = cell(1, 1);
%         R(4).prop_targets(j).fcov(3) = R(4).prop_targets(j).fcov(1);
%         R(4).prop_targets(j).fcov(1) = cell(1, 1);
%     end
% 
%     R(2) = R(2).pre_prediction(3, dt, 5 + dt); 
%     R(4) = R(4).pre_prediction(3, dt, 5 + dt);
%     [R(4), R(2)] = R(4).converge(R(2), 3);
    
    
    [R(2), R(1)] = R(2).planPath(R(1), dt, 3); % path palnning
    [R(2), R(4)] = pass_next_pos(R(2), R(4)); % data passing
    [R(4), R(3)] = R(4).planPath(R(3), dt, 3); % path palnning
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
    
    % plot the object
    for j = 1:num_target
    	plot(T(j).x, T(j).y, 'bx');
        hold on
        plot(c_t{j}(:, 1), c_t{j}(:, 2), 'r');
    end
    axis([-5,25, -5, 25])
    
    % plot the prediction
    for j = 1:num_robot/2
        for k = 1:num_target
            tmp = R(2*j).prop_targets(k).predicted;
            if ~isempty(tmp)
                plot(tmp(:,1),tmp(:,2),'rx')
            end
        end
    end
    
    % plot the robot
    for j = 1:num_robot
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.5*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.5*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.05)
end

%%
figure
for i = 1:size(a, 2)
    % plot the target
    for j = 1:num_target
        plot(c_t{j}(1:i, 1), c_t{j}(1:i, 2), 'r');
        hold on
        plot(p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b')
    end
    axis([-5, 25, -5, 25])
    
    % plot the prediction
    for j = 1:num_robot / 2
        for k = 1:num_target
            tmp = p{2*j}{k}{i};
            plot(tmp(:,1),tmp(:,2),'rx')
        end
    end
    
    % plot the robot
    for j = 1:num_robot
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.5*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.5*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.05)
end
