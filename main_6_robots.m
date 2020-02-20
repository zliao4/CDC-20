clc
clear
close all

dbstop if error

% define intial state
O1 = TargetObj(1, 2.5, 5, pi, 0.1, pi/50, 0);
R = {};
for j = 1:6
    R = [R, Robot(j, [rand*5, rand*5, rand*2*pi, 0], [0, 0])];
end
dt = 0.2;

p_r = {[],[],[],[],[],[]};
p_t = [];

for i = 1:250
    % object move
    O1 = O1.move(dt);
    p_t = [p_t; O1.x, O1.y, O1.theta];
    
    for j = 1:6
        R(j) = R(j).move(dt); % robot move
        R(j) = R(j).measure(O1); % robot measure
        p_r{j} = [p_r{j}; R(j).state, R(j).ctr]; % store position information
    end
end

% data passing
for j = 1:3
    [R(2*j), R(2*j-1)] = R(2*j).addData(R(2*j-1));
end

for i = 1:125
    for j = 1:3
        R(2*j-1) = R(2*j-1).learnGP;  % GP learning
        R(2*j-1) = R(2*j-1).prediction(2, dt);  % make prediction
        R(2*j) = R(2*j).learnGP;  % GP learning
        R(2*j) = R(2*j).prediction(2, dt);  % make prediction
    end
    
    [R(2), R(1)] = R(2).planPath2(R(1), dt); % path palnning
    [R(2), R(4)] = pass_next_pos(R(2), R(4)); % data passing
    [R(4), R(3)] = R(4).planPath2(R(3), dt); % path palnning
    [R(4), R(6)] = pass_next_pos(R(4), R(6)); % data passing
    [R(6), R(5)] = R(6).planPath2(R(5), dt); % path palnning
    
    % object move
    O1 = O1.move(dt);
    p_t = [p_t; O1.x, O1.y, O1.theta];

    for j = 1:6
        R(j) = R(j).move(dt); % robot move
        R(j) = R(j).measure(O1); % robot measure
        p_r{j} = [p_r{j}; R(j).state, R(j).ctr]; % store position information
    end
    
    % data passing
    for j = 1:3
        [R(2*j), R(2*j-1)] = R(2*j).addData(R(2*j-1));
    end
end

% plot the positions
figure    
colors = ['r', 'g', 'c', 'k', 'y', 'm'];
for i = 1:size(p_t, 1)
    plot(p_t(i,1), p_t(i,2), 'bo')
    axis([-1, 10, -1, 8])
    hold on
    plot([p_t(i,1), p_t(i,1) + 0.1*cos(p_t(i,3))], [p_t(i,2), p_t(i,2) + 0.1*sin(p_t(i,3))], 'b')
    for j = 1:6
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.1*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.1*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.05)
end