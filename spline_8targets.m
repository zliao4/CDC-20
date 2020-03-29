clc;
clear;
close all;

dbstop if error

%%
% generate the control of the target
num_target = 8;
num_data = 400;
x = cell(1, num_target);
y = cell(1, num_target);
theta = cell(1, num_target);
x{1} = 0:0.05:20;
y{1} = x{1}.^3 * 139/6600 - x{1}.^2 *1393/2200 + 1729/330*x{1};
x{2} = 21:-0.05:1;
y{2} = x{2}.^3 * 3/250 - x{2}.^2 *9/25 + 29/10*x{2};
y{3} = 18:-0.04:2;
x{3} = 97/3360 * y{3}.^3 - 473/560 * y{3}.^2 + 2629/420 * y{3} + 22/35;
x{4} = 1:0.04:18;
y{4} = -209/6120 * x{4}.^3 + 6419/6120 * x{4}.^2 - 3153/340 * x{4} + 2317/85; 
y{5} = 19:-0.0225:10;
x{5} = 10/81 * y{5}.^3 - 145/27 * y{5}.^2 + 2068/27 * y{5} -27892/81;
x{6} = 4:0.04:20;
y{6} = 1/714 * x{6}.^3 - 5/119 * x{6}.^2 - 157/714 * x{6} + 20;
x{7} = 5:0.0375:20;
y{7} = -2/75 * x{7}.^3 + 51/50 * x{7}.^2 - 349/30 * x{7} + 48;
y{8} = 2:0.03:14;
x{8} = -1/24 * y{8}.^3 + 7/6 * y{8}.^2 - 29/3 * y{8} +35;


theta{1} = atan2(x{1}.^2*139/2200-x{1}*1393/1100+1729/330, 1);
theta{2} = -atan2(x{2}.^2*9/250-x{2}*18/25+29/10, -1);
theta{3} = -atan2(97/1120*y{3}.^2-473/280*y{3}+2629/420, 1) - pi / 2;
theta{4} = atan2(-209/2040*x{4}.^2+6419/3060*x{4}-3153/340, 1);
theta{5} = -atan2(10/27*y{5}.^2-290/27*y{5}+2068/27, 1)- pi / 2;
theta{6} = atan2(3/714*x{6}.^2-10/119*x{6}-157/714, 1);
theta{7} = atan2(-2/25*x{7}.^2+51/25*x{7}-349/30, 1);
theta{8} = -atan2(-1/8*y{8}.^2+7/3*y{8}-29/3, 1) + pi / 2;


dt = 0.2;
A = cell(1, num_target);
W = cell(1, num_target);
for i = 1: num_target
    v = [0, sqrt((y{i}(2:end)-y{i}(1:end-1)).^2 + (x{i}(2:end)-x{i}(1:end-1)).^2)/dt];
    a = (v(2:end)-v(1:end-1))/dt;

    w = [0, (theta{i}(2:end)-theta{i}(1:end-1)) / dt];
    w = w(1:end-1);
    
    A{i} = a;
    W{i} = w;
end

% generate object
T = [];
for i = 1:num_target
    t = TargetObj(i, [x{i}(1)-0.25; x{i}(1)-0.25; x{i}(1)+0.25; x{i}(1)+0.25], [y{i}(1)-0.25; y{i}(1)+0.25; y{i}(1)-0.25; y{i}(1)+0.25], theta{i}(1), 0, 0, 0);
    T = [T, t];
end

% generate robot
R = [];
num_robot = 4;
for j = 1:4
    R = [R, Robot(j, [-rand*20, -rand*20, rand * 2 * pi, 0], [0, 0], num_target)];
end

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

    %%%%%% Generate new conjugacy model %%%%%%%%%%%%
    %%%%%% R2 conjugate with sigma^2 * I %%%%%%%%%%%
    %%%%%% Store pre-conju and post-conju %%%%%%%%%%
    R(2) = R(2).conjugate(3);

    R(2) = R(2).post_prediction(3, dt); 
    [R(2), R(1)] = R(2).planPath(R(1), dt, 3); % path palnning
    [R(2), R(4)] = pass_next_pos(R(2), R(4)); % data passing

    %%%%%% R4 conjugate with R2 passed pose %%%%%%%%
    %%%%%% R4 conjugate with sigma ^2 * I %%%%%%%%%%
    %%%%%% Store pre-conju and post-conju %%%%%%%%%%
    R(4) = R(4).conjugate(3);

    R(4) = R(4).post_prediction(3, dt);
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