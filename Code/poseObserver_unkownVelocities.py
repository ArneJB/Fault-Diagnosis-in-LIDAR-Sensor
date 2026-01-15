import numpy as np
import math

def R(psi: float) -> np.ndarray:
    """
    Rotation matrix from BODY frame velocities [u,v,r] to EARTH/NED rates [x_dot,y_dot,psi_dot].
    """
    c, s = math.cos(psi), math.sin(psi)
    return np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

def wrap_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


class PoseVelObserver:
    """
    Nonlinear kinematic observer using GNSS (x,y) + gyrocompass (psi) only.

    States (estimates):
        x_hat, y_hat, psi_hat    : pose (pixels, pixels, rad)
        u_hat, v_hat, r_hat      : body-fixed velocities (m/s, m/s, rad/s)

    Measurements:
        x_meas, y_meas           : GNSS position (pixels)
        psi_meas                 : gyrocompass heading (rad)

    Model:
        eta_dot = R(psi) * u
        u_dot   = 0   (unknown slowly varying)  -> estimated using innovation-driven adaptation

    Observer:
        \dot{hat eta} = R(hat psi) hat u + L_eta (y - hat y)
        \dot{hat u}   = L_u R^T(hat psi) (y - hat y)
    """

    def __init__(self, x0, y0, psi0, ppm,
                 u0=0.0, v0=0.0, r0=0.0,
                 k_eta=(0.1, 0.1, 0.9),   # gains for pose correction (x,y,psi)
                 k_u=(0.03, 0.03, 0.14)): # gains for velocity adaptation (u,v,r)
        # Pose estimates (x,y in pixels, psi in rad)
        self.x_hat = float(x0)
        self.y_hat = float(y0)
        self.psi_hat = float(psi0)

        # Velocity estimates (body frame; u,v in m/s, r in rad/s)
        self.u_hat = float(u0)
        self.v_hat = float(v0)
        self.r_hat = float(r0)

        # Pixels-per-meter
        self.ppm = float(ppm)

        # Gains
        self.kx, self.ky, self.kpsi = map(float, k_eta)
        self.ku, self.kv, self.kr = map(float, k_u)

    def predict(self, dt: float) -> None:
        """
        Prediction step: propagate pose using current velocity estimates.
        Velocity estimates follow random-walk (no deterministic prediction).
        """
        nu_hat = np.array([self.u_hat, self.v_hat, self.r_hat], dtype=float)
        eta_dot_hat = R(self.psi_hat) @ nu_hat  # [m/s, m/s, rad/s]

        # Integrate pose (x,y are pixels -> multiply meters by ppm)
        self.x_hat += (eta_dot_hat[0] * dt) * self.ppm
        self.y_hat += (eta_dot_hat[1] * dt) * self.ppm
        self.psi_hat = wrap_pi(self.psi_hat + eta_dot_hat[2] * dt)

    def correct(self, dt: float, x_meas: float, y_meas: float, psi_meas: float) -> None:
        """
        Correction step:
          - pose correction:     hat eta += K_eta * (y - hat y)
          - velocity adaptation: hat u   += K_u   * R^T(hat psi) * (y - hat y)  (discrete-time scaled by 1/dt)
        """
        # Innovation in measurement space (earth frame for position; angle wrapped)
        ex = float(x_meas) - self.x_hat
        ey = float(y_meas) - self.y_hat
        epsi = wrap_pi(float(psi_meas) - self.psi_hat)

        # --- Pose correction (innovation injection into eta) ---
        self.x_hat += self.kx * ex
        self.y_hat += self.ky * ey
        self.psi_hat = wrap_pi(self.psi_hat + self.kpsi * epsi)

        # --- Velocity adaptation (innovation injection into u) ---
        # Convert position innovation from pixels -> meters
        e_n = np.array([ex, ey, epsi], dtype=float)
        e_n[0:2] /= self.ppm  # meters, meters, rad

        # Rotate innovation to body frame (note: yaw error stays yaw error with this R^T)
        e_b = (R(self.psi_hat).T @ e_n)  # [m, m, rad]

        # Discrete-time approximation to \dot{hat u} = L_u R^T(...) (y-hat y)     
        self.u_hat += self.ku * (e_b[0] / dt)
        self.v_hat += self.kv * (e_b[1] / dt)
        self.r_hat += self.kr * (e_b[2] / dt)

    def step(self, dt: float, x_meas: float, y_meas: float, psi_meas: float):
        """
        Full observer update for one time step.
        Returns:
            x_hat, y_hat, psi_hat, u_hat, v_hat, r_hat
        """
        self.predict(dt)
        self.correct(dt, x_meas, y_meas, psi_meas)
        return self.x_hat, self.y_hat, self.psi_hat, self.u_hat, self.v_hat, self.r_hat
