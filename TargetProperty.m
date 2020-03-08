classdef TargetProperty
    properties
        % measurements
        obs;  %observation
        new_obs;  %new_observation
        num_keypoints; % the number of keypoints
        
        % GP model parameters
        hyper_param_x;
        hyper_param_y;
        GP_Model = cell(1, 2);
        model;
        
        % prediction
        predicted;
        fcov = cell(3, 1);
    end
end
        
        
        