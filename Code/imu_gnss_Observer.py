import numpy as np
import math

def R(psi):
    """Rotation matrix from body frame (u,v,r) to earth frame (x_dot,y_dot,psi_dot)."""
    return np.array([
        [math.cos(psi), -math.sin(psi), 0],
        [math.sin(psi),  math.cos(psi), 0],
        [0,              0,             1]
    ])


def wrap_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class gnssImuObserver:
    """
    State:   x_hat, y_hat, psi_hat in rad
    Inputs:  u, v, r           - body velocities
             x_meas, y_meas    - GPS measurements
             psi_meas          - gyro measurement (rad, same convention as psi_hat)
    """

    def __init__(self, x0, y0, psi0, ppm, u0, v0, k):
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

        self.psi_pred = 0
        self.previous_psi = 0

        # Observer gains (tune these)
        self.kx = k[0]
        self.ky = k[1]
        self.kpsi = k[2]

        self.ku = k[3]
        self.kv = k[4]
        self.kr = k[5]

    def predict(self, dt, u_meassured, v_meassured,r_meassured):
        """
        Prediction step: rate of change in pose based of IMU meassurements.
        """
        self.u_hat = self.u_hat + self.ku * (u_meassured - self.u_hat)
        self.v_hat = self.v_hat + self.kv * (v_meassured - self.v_hat)
        self.r_hat = self.r_hat + self.kr * (r_meassured - self.r_hat)

        nu = np.array([self.u_hat,  self.v_hat, self.r_hat])
        eta_dot = R(self.psi_hat) @ nu  # [x_dot[m/s], y_dot[m/s], psi_dot[rad/s]]

        # Position is in pixels → multiply by ppm
        self.x_hat += (eta_dot[0] * dt) * self.ppm
        self.y_hat += (eta_dot[1] * dt) * self.ppm
        self.psi_hat += eta_dot[2] * dt
        self.psi_hat = wrap_pi(self.psi_hat)

    def correctHeading(self,cog,u,v):
        """
        Heading correction with GNSS.
        """
        if cog is None:
            print("NONE")
            return
        else:
          psi_meassured = wrap_pi(cog - math.atan2(v,u)) #Course Heading - Side slip angle
          if u < 0:
            psi_meassured = wrap_pi(psi_meassured + math.pi)
          #e_psi = wrap_pi(psi_meassured - self.psi_hat_prev)
          #psi_hat = wrap_pi(self.psi_hat + self.kpsi * e_psi)
          e_psi = wrap_pi(psi_meassured - self.psi_hat)
          self.psi_hat = wrap_pi(self.psi_hat + self.kpsi * e_psi)
          self.psi_hat_prev = self.psi_hat    

    def correct(self, x_meas, y_meas,dt):
        """
        Position correction with GNSS.
        """
        # Integrate
        #x_pred = self.x_hat + (self.dx_pred * dt) * self.ppm
        #y_pred = self.y_hat + (self.dy_pred * dt) * self.ppm
        
        # Position correction
        self.x_hat = self.x_hat  + self.kx * (x_meas - self.x_hat)
        self.y_hat = self.y_hat  + self.ky * (y_meas - self.y_hat)

    def step(self, dt, u, v, r, x_meas, y_meas, cog):
        """
        Full observer update for one time step.
        Returns (x_hat, y_hat, psi_hat).
        """
        self.predict(dt ,u, v, r)
        #self.correctHeading(cog,u, v)
        self.correct(x_meas, y_meas,dt)
        return self.x_hat, self.y_hat, self.psi_hat, self.r_hat