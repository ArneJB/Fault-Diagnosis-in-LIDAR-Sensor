import numpy as np

class IMU:
    """
    Simple IMU that measures body velocities nu = [u, v, r]^T
    with Gaussian noise and optional constant bias.
    Units must match your model (m/s, rad/s).
    """

    def __init__(self, sigma_nu):
        """
        sigma_nu: [sigma_u, sigma_v, sigma_r]
        """
        self.sigma_nu = np.array(sigma_nu, dtype=float)

    def measure(self, nu_true,fault_var,fault_bias):
        """
        nu_true: [u, v, r] from the simulator
        returns: noisy measurement nu_meas
        """
        nu_true = np.array(nu_true, dtype=float)
        noise   = np.random.normal(0.0, self.sigma_nu+fault_var, size=3)
        nu_meas = nu_true + noise + fault_bias
        return nu_meas  # [u_meas, v_meas, r_meas]