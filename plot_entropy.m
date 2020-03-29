function plot_entropy(e, num_data)
    figure
    
    legend
    for i =1: size(e, 2)
        plot(1:num_data, e{i}, 'LineWidth', 2);
        hold on
    end
    
    font_size = 40;
    th = title('Information entropy','FontSize', font_size);
    set(th, 'Interpreter', 'none')
    xAX = get(gca,'XAxis');
    set(xAX,'FontSize', font_size)
    yAX = get(gca,'yAxis');
    set(yAX,'FontSize', font_size)
    xlabel('Time Step')
    ylabel('Prediction Error')
    box on
    leg = legend('Decentralized data fusion','Random control','Target pursuit');
    leg.FontSize = font_size - 10;
end
