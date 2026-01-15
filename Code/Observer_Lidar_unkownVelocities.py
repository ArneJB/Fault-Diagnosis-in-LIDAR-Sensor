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

class lidar_gyro_observer:
    """
    State:   x_hat, y_hat, psi_hat in rad
    Inputs: 
    """

    def __init__(self, x0, y0, psi0, ppm, k,LandMarkPose,radius):#,ly,lx,r):
        # Estimated state
        self.x_hat = x0
        self.y_hat = y0
        self.psi_hat = psi0

        self.u_hat = 0
        self.v_hat = 0
        self.r_hat = 0
        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        self.dx_pred = 0
        self.dy_pred = 0

        self.psi_hat_prev = 0

        # Observer gains
        self.ku = k[0]
        self.kv = k[1]
        self.kr = k[2]

        self.kpsi = k[3]
        self.k_lidar = k[4]


        self.lx = LandMarkPose[0]
        self.ly = LandMarkPose[1]
        self.radius = radius
        self.l_meassured = 0

    def predict(self, dt, sensor_data):
        """
        Prediction step: rate of change in pose based of estimated velocities and GNSS.
        """
        nu_hat = np.array([self.u_hat, self.v_hat, self.r_hat])
        eta_dot_hat = R(self.psi_hat) @ nu_hat  # [x_dot[m/s], y_dot[m/s], psi_dot[rad/s]]

        # Position is in pixels → multiply by ppm
        self.x_hat += (eta_dot_hat[0] * dt) * self.ppm
        self.y_hat += (eta_dot_hat[1] * dt) * self.ppm
        self.psi_hat = wrap_pi(self.psi_hat + eta_dot_hat[2] * dt)


        zi_global = point_matching.pointCloud_screen_global(sensor_data,[self.x_hat,self.y_hat,self.psi_hat])
        associations = point_matching.match_points([(self.lx,self.ly)],zi_global,15,self.radius,self.ppm)
        self.l_meassured = point_matching.meassured_circle(associations,self.radius)
        


    def correct(self, psi_meas,dt):
        """
        Correction step: Meassured pose of GPS vs estimated.
        """
        # Innovation errors
        epsi = psi_meas - self.psi_hat
        e_lidar = [self.lx,self.ly] - self.l_meassured
                
        # Corrections
        self.psi_hat = wrap_pi(self.psi_hat + self.kpsi * epsi)
        #X and Y are a meassures of pixels
        self.x_hat = self.x_hat + self.k_lidar * e_lidar[0] 
        self.y_hat = self.y_hat + self.k_lidar * e_lidar[1]

        # Velocity Observer
        # Convert position innovation from pixels to meters
        e_vel = np.array([e_lidar[0], e_lidar[1], epsi], dtype=float)
        e_vel[0:2] /= self.ppm  # meters, meters, rad
 
        e_b = (R(self.psi_hat).T @ e_vel)

        self.u_hat += self.ku * (e_b[0])
        self.v_hat += self.kv * (e_b[1])
        self.r_hat += self.kr * (e_b[2])

    def step(self, dt,sensor_data,psi_meas):
        """
        Full observer update for one time step.
        Returns (x_hat, y_hat, psi_hat).
        """
        self.predict(dt,sensor_data)
        self.correct(psi_meas,dt)

        dx = self.lx - self.x_hat
        dy = self.ly - self.y_hat
        d = math.sqrt(dx*dx + dy*dy)/self.ppm
        return self.x_hat, self.y_hat, self.psi_hat, d