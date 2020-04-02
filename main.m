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
y{6} = 1/307 * x{6}.^3 - 10/119 * x{6}.^2 - 157/307* x{6} + 20;
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
    R = [R, Robot(j, [rand * 20, rand * 20, rand * 2 * pi, 0], [0, 0], num_target)];
end

%% run three different control policy

[p_t1, c_t1, p1, p_r1, e1] = spline_5_targets_info(A, W, T, num_target, R, num_robot, num_data, dt);
[p_t2, c_t2, p2, p_r2, e2] = spline_5_targets_random(A, W, T, num_target, R, num_robot, num_data, dt);
[p_t3, c_t3, p3, p_r3, e3] = spline_5_targets_pursue_nearest(A, W, T, num_target, R, num_robot, num_data, dt);
[p_t4, c_t4, p4, p_r4, e4] = spline_5_targets_optimal(A, W, T, num_target, R, num_robot, num_data, dt);
%% plot result

plot_path({c_t1, c_t2, c_t3, c_t4}, {p_t1, p_t2, p_t3, p_t4}, {p1, p2, p3, p4}, {p_r1, p_r2, p_r3, p_r4}, num_data, num_target, num_robot)
plot_error({p1, p4, p3, p2}, {p_t1, p_t4, p_t3, p_t2}, num_data, num_target, num_robot, 5)
plot_entropy({e1,e2,e3,e4}, num_data)
