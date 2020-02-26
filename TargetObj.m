classdef TargetObj
    properties
        % state
        idx;
        x;
        y;
        theta;
        v;
        
        % size of key point
        s;
        
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
            this.s = size(x, 1);
        end
        
        function this = move(this, dt)
            % mass center
            cx = mean(this.x);
            cy = mean(this.y);  
            
            % mass center new position
            dx = this.v * cos(this.theta) * dt;
            dy = this.v * sin(this.theta) * dt;
            
            % compute the new position of every key point
            new_x = []; new_y = [];
            for i = 1:this.s
                d = norm([this.x(i) - cx, this.y(i) - cy]);  % distance to center
                theta = atan2(this.y(i) - cy, this.x(i) - cx);
                new_x = [new_x; this.x(i) + dx + d*(cos(theta+this.w*dt) - cos(theta))];
                new_y = [new_y; this.y(i) + dy + d*(sin(theta+this.w*dt) - sin(theta))];
            end
            cx = cx + dx;
            cy = cy + dy;
            
            this.fx = new_x - this.x;
            this.fy = new_y - this.y;
            this.x = new_x;
            this.y = new_y;
            this.theta = this.theta + this.w * dt;
            this.v = this.v + this.a * dt;
            this.t = this.t + dt;
        end
    end
end