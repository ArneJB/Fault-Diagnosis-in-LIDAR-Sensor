import numpy as np
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
    """
    State:   x_hat, y_hat, psi_hat in rad
    Inputs:  u, v, r           - estimated body velocities
             x_meas, y_meas    - GPS measurements
    """

    def __init__(self, x0, y0, psi0, ppm, u0, v0, k):#,ly,lx,r):
        # Estimated state
        self.x_hat = x0
        self.y_hat = y0
        self.psi_hat = psi0

        self.u = u0
        self.v = v0
        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        self.dx_pred = 0
        self.dy_pred = 0

        self.previous_psi = 0

        #self.ly = ly
        #self.lx = lx
        #self.r = r

        # Observer gains
        self.kx = k[0]
        self.ky = k[1]

    def predict(self, dt, u_estimated, v_estimated,r_estimated):
        """
        Prediction step: rate of change in pose based of estimated velocities and GNSS.
        """
        nu = np.array([u_estimated, v_estimated, r_estimated])
        eta_dot = R(self.psi_hat) @ nu  # [x_dot[m/s], y_dot[m/s], psi_dot[rad/s]]

        # Position is in pixels → multiply by ppm
        self.x_hat += (eta_dot[0] * dt) * self.ppm
        self.y_hat += (eta_dot[1] * dt) * self.ppm
        self.psi_hat += eta_dot[2] * dt
        self.psi_hat = wrap_pi(self.psi_hat)



        # self.psi_hat = wrap_pi(self.psi_hat + r_estimated * dt) #discrete integration of r
        
        # self.dx_pred = u_estimated * math.cos(self.psi_hat) - v_estimated * math.sin(self.psi_hat)
        # self.dy_pred = u_estimated * math.sin(self.psi_hat) + v_estimated * math.cos(self.psi_hat)

    def correct(self, x_meas, y_meas,dt):
        """
        Correction step: Meassured pose of GPS vs estimated.
        """
        # Integrate
        #x_pred = self.x_hat + (self.dx_pred * dt) * self.ppm
        #y_pred = self.y_hat + (self.dy_pred * dt) * self.ppm
        
        # Position correction
        self.x_hat = self.x_hat + self.kx * (x_meas - self.x_hat)
        self.y_hat = self.y_hat + self.ky * (y_meas - self.y_hat)

    def step(self, dt, u, v, r, x_meas, y_meas):
        """
        Full observer update for one time step.
        Returns (x_hat, y_hat, psi_hat).
        """
        self.predict(dt ,u, v, r)
        self.correct(x_meas, y_meas,dt)
        return self.x_hat, self.y_hat, self.psi_hat