function plot_path(C_t, P_t, P, P_r, num_data, num_target, num_robot)  
    
    for i = 1:num_data
        for m = 1:size(C_t, 2) 
            figure(m)
            c_t = C_t{m}; p_t = P_t{m}; p = P{m}; p_r = P_r{m};
            colors = {'r', 'g', 'm', 'k', 'b', [0,90/256,154/256], '#EDB120' ,'#77AC30'};
            
            % plot the target
            for j = 1:num_target
                plot(c_t{j}(1:i, 1), c_t{j}(1:i, 2), '-.', 'Color' ,colors{j},  'LineWidth', 2);
                hold on
%                 plot(p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b',  'LineWidth', 2)
            
                plot(2*p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1)- c_t{j}(i,1), 2*p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2)-c_t{j}(i,2), 'b',  'LineWidth', 2)
            end
            axis([-5, 25, -5, 25])

            %plot the prediction
%             for k = 1:num_target
%                 tmp_pre = [];
%                 for j = 1:num_robot / 2
%                     tmp = p{2*j}{k}{i};
%                     if size(tmp, 1) > 1
%                         tmp_pre = [tmp_pre; mean(tmp(1:4,1)),mean(tmp(1:4,2))];
%                     end
%                 end
%                 plot(mean(tmp_pre(:, 1)), mean(tmp_pre(:, 2)),'Color' ,'r','Marker', 'x', 'MarkerSize',16, 'LineWidth', 2)
%             end
                    

            % plot the robot
            for j = 1:num_robot
                tmp_p = p_r{j};
                plot(tmp_p(i,1), tmp_p(i,2), 'Color', colors{j}, 'Marker', 'o', 'MarkerSize',12, 'LineWidth', 2)
                plot([tmp_p(i,1), tmp_p(i,1) + 1/3*cos(tmp_p(i,3))], [tmp_p(i,2), tmp_p(i,2) + 1/3*sin(tmp_p(i,3))],'Color',colors{j}, 'LineWidth', 2)
            end
            hold off
        end
        pause(0.05)
    end
    
font_size = 20;
th = title('Initial layout','FontSize', font_size);
set(th, 'Interpreter', 'none')
xAX = get(gca,'XAxis');
set(xAX,'FontSize', font_size)
yAX = get(gca,'yAxis');
set(yAX,'FontSize', font_size)
xlabel('X / m')
ylabel('Y / m')
box on
leg = legend;
leg.FontSize = font_size;
end