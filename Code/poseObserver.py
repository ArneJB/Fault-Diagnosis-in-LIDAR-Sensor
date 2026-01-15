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


class PoseObserver:
    """
    Simple nonlinear observer / complementary filter for vessel pose.

    State:   x_hat, y_hat, psi_hat in rad

    Inputs:  u, v, r           - body velocities (ASSUMED KNOWN)
             x_meas, y_meas    - GPS measurements
             psi_meas          - gyro measurement (rad, same convention as psi_hat)
    """

    def __init__(self, x0, y0, psi0, ppm,
                 k):
        # Estimated state
        self.x_hat = x0
        self.y_hat = y0
        self.psi_hat = psi0

        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        # Observer gains (tune these)
        self.kx = k[0]
        self.ky = k[1]
        self.kpsi = k[2]

    def predict(self, dt, u, v, r):
        """
        Prediction step: integrate kinematics with current estimate and inputs.
        """
        nu = np.array([u, v, r])
        eta_dot = R(self.psi_hat) @ nu  # [x_dot[m/s], y_dot[m/s], psi_dot[rad/s]]

        # Position is in pixels → multiply by ppm
        self.x_hat += (eta_dot[0] * dt) * self.ppm
        self.y_hat += (eta_dot[1] * dt) * self.ppm
        self.psi_hat += eta_dot[2] * dt
        self.psi_hat = wrap_pi(self.psi_hat)

    def correct(self, x_meas, y_meas, psi_meas):
        """
        Correction step: move estimate towards measurements.
        """
        # Position correction
        self.x_hat += self.kx * (x_meas - self.x_hat)
        self.y_hat += self.ky * (y_meas - self.y_hat)

        # Heading correction – be careful with wrap-around
        d_psi = wrap_pi(psi_meas - self.psi_hat)
        self.psi_hat += self.kpsi * d_psi
        self.psi_hat = wrap_pi(self.psi_hat)

    def step(self, dt, u, v, r, x_meas, y_meas, psi_meas):
        """
        Full observer update for one time step.
        Returns (x_hat, y_hat, psi_hat).
        """
        self.predict(dt, u, v, r)
        self.correct(x_meas, y_meas, psi_meas)
        return self.x_hat, self.y_hat, self.psi_hat