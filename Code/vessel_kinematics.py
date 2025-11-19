import math
import numpy as np

def R(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])

class marineVessel:
    def __init__(self,x,y,psi):
        #Initial conditions passed to constructor
        self.x = x
        self.y = y
        self.psi = psi

        #Start with 0 velocities until others are specified
        self.u = 0 #Surge Velocity
        self.v = 0 #Sway Velocity
        self.r = 0 #Yaw rate

        self.nu = np.array([self.u,self.v,self.r])      #Fixed body velocities
        self.eta_dot = R(self.psi) @ self.nu       #Global Earth frame rates

    def setVelocities(self,u, v ,r):
        self.u = u
        self.v = v
        self.r = r
        self.nu = np.array([self.u,self.v,self.r])      #Fixed body velocities

    #variables are sample time + pixel pr meter
    def updatePose(self,dt,ppm):
        self.eta_dot = R(self.psi) @ self.nu

        #Eurler Integration
        self.x   += (self.eta_dot[0] * dt) * ppm # Transform into global frame with pixel per meter scaling
        self.y   += (self.eta_dot[1] * dt) * ppm # Transform into global frame with pixel per meter scaling
        self.psi += (self.eta_dot[2] * dt)       # rad/s is not affected by pixel coordinates

        return (self.x,self.y,self.psi)
        
