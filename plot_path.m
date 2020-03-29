function plot_path(C_t, P_t, P, P_r, num_data, num_target, num_robot)  
    
    for i = 1:num_data
        for m = 1:size(C_t, 2) 
            figure(m)
            c_t = C_t{m}; p_t = P_t{m}; p = P{m}; p_r = P_r{m};
            colors = ['r', 'g', 'c', 'k', 'y', 'm'];
            
            % plot the target
            for j = 1:num_target
                plot(c_t{j}(1:i, 1), c_t{j}(1:i, 2), 'r');
                hold on
                plot(p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],1), p_t{j}([4*i-3,4*i-2,4*i,4*i-1,4*i-3],2), 'b')
            end
            axis([-5, 25, -5, 25])

            %plot the prediction
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
        end
        pause(0.05)
    end
end