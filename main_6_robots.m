
clc
clear
close all

dbstop if error

% % define intial state
O1 = TargetObj(1, [10; 10; 10.5; 10.5], [20; 19.5; 20; 19.5], 0, pi/2, -pi/20, 0);
O2 = TargetObj(2, [1; 1; 1.5; 1.5], [5; 5.5; 5; 5.5], 0, pi/3, -pi/5, 0);
R = [];
for j = 1:6
    R = [R, Robot(j, [rand*20, rand*20, rand*2*pi, 0], [0, 0])];
end
% R = [Robot(j, [5, 5, rand*2*pi, 0], [0, 0]),
%      Robot(j, [6, 5, rand*2*pi, 0], [0, 0]),
%      Robot(j, [15, 15, rand*2*pi, 0], [0, 0]),
%      Robot(j, [17, 14, rand*2*pi, 0], [0, 0]),
%      Robot(j, [20, 10, rand*2*pi, 0], [0, 0]),
%      Robot(j, [19, 12, rand*2*pi, 0], [0, 0])];

    
dt = 0.2;

p_r = {[],[],[],[],[],[]};
p_t = [];
p_t2 = [];

for i = 1:50
    % object move
    O1 = O1.move(dt);
    O2 = O2.move(dt);
    p_t = [p_t; O1.x, O1.y, O1.theta * ones(size(O1.x))];
    p_t2 = [p_t2; O2.x, O2.y, O2.theta * ones(size(O2.x))];
    
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

for i = 1:200
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
    O2 = O2.move(dt);
    p_t = [p_t; O1.x, O1.y, O1.theta * ones(size(O1.x))];
    p_t2 = [p_t2; O2.x, O2.y, O2.theta * ones(size(O2.x))];

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
for i = 1:size(p_t, 1) / O1.s
    plot(p_t([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b')
    hold on
    axis([-5, 25, -5, 25])
    %plot(p_t2([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t2([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b')
    for j = 1:6
        tmp_p = p_r{j};
        plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors(j), 'Marker', 'o')
        plot([tmp_p(i,1), tmp_p(i,1) + 0.1*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 0.1*sin(tmp_p(i,3))], colors(j))
    end
    hold off
    pause(0.01)
end