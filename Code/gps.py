import math
import numpy as np

def addUncertainty(x,y, sigma):
    mean   = np.array([0, 0])  # zero-mean noise
    covar  = np.diag(sigma**2)
    noise = np.random.multivariate_normal(mean,covar)
    return x+noise[0], y+noise[1]


class GPS:
    def __init__(self,pose,uncertainty):
        self.x = pose[0]
        self.y = pose[1]
        self.sigma = np.array(uncertainty)

        self.prev_x = self.x
        self.prev_y = self.y
        self.fault = 0

        self.initialized = False

    def gpsPose(self,dt,pose):

        # Store previous GNSS fix
        x_prev = self.prev_x
        y_prev = self.prev_y

        self.x = pose[0] + self.fault
        self.y = pose[1] + self.fault
        self.x, self.y = addUncertainty(self.x,self.y,self.sigma)
        # Save current measurement for next COG computation
        self.prev_x = self.x
        self.prev_y = self.y

        if not self.initialized:
            self.initialized = True
            return self.x, self.y, None   # COG undefined

        dx = self.x - x_prev
        dy = self.y - y_prev

        speed = math.sqrt(dx*dx + dy*dy) / dt

        if speed < 0.2:        # adjust threshold as needed
            cog = None
        else:
            # COG
            cog = math.atan2(dy, dx)
            cog = (cog + math.pi) % (2*math.pi) - math.pi

        return self.x, self.y, cog
    def setUncertainty(self,std,fault):
        self.sigma = np.array(std)
        self.fault = fault