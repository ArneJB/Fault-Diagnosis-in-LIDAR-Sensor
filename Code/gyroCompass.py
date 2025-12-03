import math
import numpy as np

def addUncertainty(psi, sigma):
    mean   = 0  # zero-mean noise
    covar  = sigma**2
    noise = np.random.normal(mean,covar)
    return psi+noise


class gyroCompass:
    def __init__(self,psi,uncertainty):
        self.psi = psi
        self.sigma = uncertainty
    def psi_measured(self,psi,fault_bias,fault_noise):
        self.psi = psi
        self.psi = addUncertainty(self.psi,self.sigma+fault_noise)
        return self.psi + fault_bias