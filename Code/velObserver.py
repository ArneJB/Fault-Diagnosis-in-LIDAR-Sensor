import numpy as np
import math

def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi

class VelObserver:    
    def __init__(self, u0=0.0, v0=0.0, r0=0.0, psi0 = 0.0, k_u= 0.2 ,k_v=0.2, k_r=0.2, k_psi = 0.01):
        self.u_hat = u0
        self.v_hat = v0
        self.r_hat = r0
        self.psi_hat = psi0
        self.k_u   = k_u
        self.k_v   = k_v
        self.k_r   = k_r
        self.k_psi = k_psi
    def step(self, dt, u_model, v_model, r_model, u_meas, v_meas, r_meas, psi_meas):
        # Prediction from model / commanded values
        psi_pred = self.psi_hat + r_meas*dt
        psi_pred = wrap(psi_pred)

        u_pred = u_model
        v_pred = v_model
        r_pred = r_model

        # Correction with measurements
        # heading correction (compass)
        psi_err = wrap(psi_meas - psi_pred)
        psi_hat = psi_pred + self.k_psi * psi_err
        psi_hat = wrap(psi_hat)

        self.psi_hat = psi_hat

        self.u_hat = u_pred + self.k_u * (u_meas - u_pred)
        self.v_hat = v_pred + self.k_v * (v_meas - v_pred)
        self.r_hat = r_pred + self.k_r * (r_meas - r_pred)

        return self.u_hat, self.v_hat, self.r_hat, psi_hat