function plot_error(p, p_t, num_data, num_target, num_robot, future_frame)

figure
n = size(p,2);
errors = cell(1, n);
for i = 1:num_data - future_frame + 1
    for m = 1:n
        tmp_error = 0;
        for j = 1:num_robot / 2
            tmp_k = 0;
            for k = 1:num_target
                tmp = p{m}{2*j}{k}{i};
                if size(tmp, 1) ~= 1
                    tmp_error = tmp_error + mean(sum((tmp(:,1:2) - p_t{m}{k}(4*i-3:4*i+16,1:2)).^2, 2));
                    tmp_k = tmp_k + 1;
                end
            end
            tmp_error = tmp_error / tmp_k;
        end
        tmp_error = tmp_error / (num_robot / 2);
        errors{m} = [errors{m}, sqrt(tmp_error)];
    end
end

legend 
for i = 1:n
    plot(1:num_data- future_frame + 1, errors{i}, 'LineWidth', 2)
    hold on
end

font_size = 40;
th = title('GP prediction error','FontSize', font_size);
set(th, 'Interpreter', 'none')
xAX = get(gca,'XAxis');
set(xAX,'FontSize', font_size)
yAX = get(gca,'yAxis');
set(yAX,'FontSize', font_size)
xlabel('Time Step')
ylabel('Prediction Error / m')
box on
leg = legend('Decentralized data fusion','Random control','Target pursuit');
leg.FontSize = font_size - 10;


