import numpy as np
import point_matching
import math

def wrap_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def R(psi):
    """Rotation matrix from body frame (u,v,r) to earth frame (x_dot,y_dot,psi_dot)."""
    return np.array([
        [math.cos(psi), -math.sin(psi), 0],
        [math.sin(psi),  math.cos(psi), 0],
        [0,              0,             1]
    ])

class lidar_gnss_observer:
    def __init__(self, x0, y0, ppm, k,LandMarkPose,radius):#,ly,lx,r):
        # Estimated state
        self.x_hat = x0
        self.y_hat = y0

        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        # Observer gains
        self.kx = k[0]
        self.ky = k[1]
        self.kd = k[2]
        self.kxy = k[3]

        self.lx = LandMarkPose[0]
        self.ly = LandMarkPose[1]
        self.radius = radius

        self.d_pred = 0.0
        self.d_meas = 0.0
        self.d_hat = 0.0

    def predict(self,x_meassured):
        dx = (self.lx - self.x_hat) / self.ppm
        dy = (self.ly - self.y_hat) / self.ppm
        self.d_pred = math.sqrt(dx*dx + dy*dy)  # meters

    def correct(self, x_meas, y_meas,sensor_data,dt):
        # Innovation errors
        self.x_hat = self.x_hat + self.kx * (x_meas - self.x_hat)
        self.y_hat = self.y_hat + self.ky * (y_meas - self.y_hat)

        #Distance Meassured
        pointCloudBody = point_matching.pointCloud_body(sensor_data)
        associations_raw = point_matching.match_points_raw(pointCloudBody)
        c_hat_raw = point_matching.meassured_circle(associations_raw,self.radius)
        self.d_meas  = point_matching.meassured_distance((0,0),c_hat_raw,self.ppm)
       
        e_dist = self.d_meas - self.d_pred #Error distance
        self.d_hat = self.d_pred        
        # Correct Pose with Distance error as well
        dx_m = (self.lx - self.x_hat) / self.ppm
        dy_m = (self.ly - self.y_hat) / self.ppm
        d = math.sqrt(dx_m*dx_m + dy_m*dy_m)        
        e_LOS_x = dx_m / d #Direction of error in x
        e_LOS_y = dy_m / d #Direction of error in y

        self.x_hat = self.x_hat + (self.kxy * e_LOS_x * e_dist) * self.ppm
        self.y_hat = self.y_hat + (self.kxy * e_LOS_y * e_dist) * self.ppm

        
    def step(self, dt,sensor_data,x_meas,y_meas):
        self.predict(x_meas)
        self.correct(x_meas, y_meas, sensor_data, dt)
       
        return self.x_hat, self.y_hat, self.d_hat