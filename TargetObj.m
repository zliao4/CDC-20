classdef TargetObj
    properties
        % state
        idx;
        x;
        y;
        theta;
        v;
        
        % control
        w;
        a;
        
        % flow
        fx;
        fy;
        
        % time
        t;
    end
    
    methods
        function this = TargetObj(idx, x, y, theta, v ,w, a)
            this.idx = idx;
            this.x = x;
            this.y = y;
            this.theta = theta;
            this.v = v;
            this.w = w;
            this.a = a;
            this.t = 0;
        end
        
        function this = move(this, dt)
            this.fx = this.v * cos(this.theta) * dt;
            this.fy = this.v * sin(this.theta) * dt;
            this.x = this.x + this.fx;
            this.y = this.y + this.fy;
            this.theta = this.theta + this.w * dt;
            this.v = this.v + this.a * dt;
            this.t = this.t + dt;
        end
    end
end