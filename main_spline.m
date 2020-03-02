clc;
clear;
close all;

x = 0:0.05:20;
y = x.^3 * 139/6600 - x.^2 *1393/2200 + 1729/330*x;
theta = atan2(x.^2*139/2200-x*1393/1100+1729/330, 1);
dt = 0.2;
v = [0, sqrt((y(2:end)-y(1:end-1)).^2 + (x(2:end)-x(1:end-1)).^2)/dt];
w = [0, (theta(2:end)-theta(1:end-1)) / dt];
w = w(1:end-1);
a = (v(2:end)-v(1:end-1))/dt;

t = TargetObj(1, [-0.25; -0.25; 0.25; 0.25], [-0.25; 0.25; -0.25; 0.25], theta(1), 0, 0, 0);

R = [];
num_robot = 4;
for j = 1:num_robot/2
    R = [R, Robot(j, [rand*5+2.5, rand*5+10.5, rand*2*pi, 0], [0, 0])];
end

for j = num_robot/2+1:num_robot
    R = [R, Robot(j, [-rand*5+18.5, -rand*5+9.5, rand*2*pi, 0], [0, 0])];
end

% R = [R, Robot(1, [-rand*5+6, rand*5+19.5, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(2, [rand*5+30, -rand*5+14.25, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(3, [-rand*5+6, rand*5+19.5, rand*2*pi, 0], [0, 0])];
% R = [R, Robot(4, [rand*5+30, -rand*2+14.25, rand*2*pi, 0], [0, 0])];

p_t = [];
c_t = [];
p_r = {[],[],[],[],[],[]};
colors = ['r', 'g', 'c', 'k', 'y', 'm'];
p = {{},{},{},{},{},{}};
for i = 1:size(a, 2)
    % object move
    t.a = a(i);
    t.w = w(i);
    t = t.move(dt);
    p_t = [p_t; t.x, t.y, t.theta * ones(size(t.x))];
    c_t = [c_t; mean(t.x), mean(t.y), t.theta];
    
    for j = 1:num_robot
        R(j) = R(j).move(dt); % robot move
        R(j) = R(j).measure(t); % robot measure
        p_r{j} = [p_r{j}; R(j).state, R(j).ctr]; % store position information
    end
    
    % data passing
    for j = 1:num_robot / 2
        [R(2*j), R(2*j-1)] = R(2*j).addData(R(2*j-1));
    end
    
    for j = 1:num_robot/2
        R(2*j) = R(2*j).learnGP;  % GP learning
        R(2*j) = R(2*j).prediction(2, dt);  % make prediction
    end
    
    [R(2), R(1)] = R(2).planPath(R(1), dt); % path palnning
    [R(2), R(4)] = pass_next_pos(R(2), R(4)); % data passing
    [R(4), R(3)] = R(4).planPath(R(3), dt); % path palnning
    
    for j = 1:num_robot/2
        if ~isempty(R(2*j).predicted{1})
            p{2*j} = [p{2*j}, R(2*j).predicted{1}];
        else
            p{2*j} = [p{2*j}, {[0, 0]}];
        end
    end
    
    plot(t.x, t.y, 'bx')
    hold on
    
    plot(c_t(:, 1), c_t(:, 2), 'r');
    axis([-5,25, -5, 25])
    for j = 1:num_robot/2
        tmp = R(2*j).predicted{1};
        if ~isempty(tmp)
            plot(tmp(1:4,1),tmp(1:4,2),'rx')
        end
    end
    for j = 1:num_robot
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.1*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.1*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.05)
end

figure
for i = 1:size(a, 2)
    plot(c_t(1:i, 1), c_t(1:i, 2), 'r');
    hold on
    plot(p_t([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b')
    axis([-5, 25, -5, 25])
    for j = 1:num_robot / 2
        tmp = p{2*j}{i};
        plot(tmp(:,1),tmp(:,2),'rx')
    end
    for j = 1:num_robot
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.1*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.1*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.05)
end
