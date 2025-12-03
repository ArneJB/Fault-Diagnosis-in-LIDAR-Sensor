import numpy as np
import math

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

        self.u = u0
        self.v = v0
        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        self.dx_hat = 0
        self.dy_hat = 0

        self.previous_psi = 0

        # Observer gains (tune these)
        self.kx = k[0]
        self.ky = k[1]
        self.kpsi = k[2]

    def predict(self, dt, u_meassured, v_meassured,r_meassured,cog_meassured):
        """
        Prediction step: rate of change in pose based of IMU meassurements.
        """

        psi_pred = wrap_pi(self.psi_hat + r_meassured * dt) #discrete integration of r
        
        if cog_meassured is None:
            print("NONE")
            psi_meassured = self.previous_psi #If value is non vessel is moving too slow to compute
        else:
            psi_meassured = cog_meassured - math.atan2(v_meassured,u_meassured) #Course Heading - Side slip angle


        e_psi = wrap_pi(psi_meassured - psi_pred)
        psi_hat = wrap_pi(psi_pred + self.kpsi * e_psi)
        self.psi_hat = psi_hat
        
        self.dx_hat = u_meassured * math.cos(psi_hat) - v_meassured * math.sin(psi_hat)
        self.dy_hat = u_meassured * math.sin(psi_hat) + v_meassured * math.cos(psi_hat)

        self.previous_psi = psi_hat

    def correct(self, x_meas, y_meas,dt):
        """
        Correction step: Meassured pose of GPS vs estimated.
        """
        # Integrate
        self.x_hat += (self.dx_hat * dt) * self.ppm
        self.y_hat += (self.dy_hat * dt) * self.ppm
        
        # Position correction
        self.x_hat += self.kx *(x_meas - self.x_hat)
        self.y_hat += self.ky * (y_meas - self.y_hat)

    def step(self, dt, u, v, r, x_meas, y_meas, cog):
        """
        Full observer update for one time step.
        Returns (x_hat, y_hat, psi_hat).
        """
        self.predict(dt ,u, v, r, cog)
        self.correct(x_meas, y_meas,dt)
        return self.x_hat, self.y_hat, self.previous_psi