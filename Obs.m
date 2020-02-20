classdef Obs
    properties
        % position
        x;
        y;
        
        % flow
        fx;
        fy;
        
        % time
        t;
    end
    
    methods
        function this = Obs(x, y, fx, fy, t)
            this.x = x;
            this.y = y;
            this.fx = fx;
            this.fy = fy;
            this.t = t;
        end
    end
end
