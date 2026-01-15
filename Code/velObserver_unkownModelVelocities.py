import numpy as np
import math

def wrap(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi

class VelObserver:    
    def __init__(self, u0=0.0, v0=0.0, r0=0.0, psi0 = 0.0, k_u= 0.2 ,k_v=0.2, k_r=0.2, k_psi = 0.2):
        self.u_hat = u0
        self.v_hat = v0
        self.r_hat = r0
        self.psi_hat = psi0
        self.k_u   = k_u
        self.k_v   = k_v
        self.k_r   = k_r
        self.k_psi = k_psi
    def step(self, dt, u_meas, v_meas, r_meas, psi_meas):


        self.r_hat = self.r_hat + self.k_r * (r_meas - self.r_hat)

        # Prediction - tie imu and gyro together with r*dt
        psi_pred = wrap(self.psi_hat + self.r_hat*dt)

        # Correction with measurements
        # heading correction (compass)
        psi_err = wrap(psi_meas - psi_pred)
        self.psi_hat = wrap(psi_pred + self.k_psi * psi_err)

        self.u_hat = self.u_hat + self.k_u * (u_meas - self.u_hat)
        self.v_hat = self.v_hat + self.k_v * (v_meas - self.v_hat)

        return self.u_hat, self.v_hat, self.r_hat, self.psi_hat