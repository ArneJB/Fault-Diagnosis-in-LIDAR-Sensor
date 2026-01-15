import numpy as np
import math
import point_matching
from scipy import signal

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


class structuralResidual:
    """
    Class to genreate residuals from structural analysis
    y1: GNSS
    y2: IMU
    y3: Gyrocompass
    y4: LiDAR
    """
    def __init__(self, y1, y2, y3,y4,l,radius,ppm):
        self.y1 = y1
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4

        self.y1_previous = [0.0, 0.0]
        self.y2_previous = [0.0, 0.0, 0.0]
        self.y3_previous = 0.0

        self.lx = l[0]
        self.ly = l[1]
        self.radius = radius
        # Pixels-per-meter (for consistency with your kinematics)
        self.ppm = ppm

        self.r1 = []
        self.r2 = []
        self.r3 = []
        self.initialize = 0

        #Filte parameters
        self.w0 = 2*np.pi*2     # rad/s  (0.5 Hz cutoff can be tuned)
        self.zeta = 0.7 

        self.num_c = [self.w0**2]
        self.den_c = [1, 2*self.zeta*self.w0, self.w0**2]
        dt = 1/60#0.33333
        # Discretize filter using bilinear 
        num_z, den_z, _ = signal.cont2discrete((self.num_c, self.den_c), dt, method="bilinear")
        num_z = num_z.squeeze()  # cont2discrete gives shape (1, N)
        den_z = den_z.squeeze()

        self.b, self.a = num_z, den_z
        #save to dimensions
        nstate = max(len(self.a), len(self.b)) - 1
        self.zi_r1 = np.zeros(nstate)        # r1 is scalar
        self.zi_r2 = np.zeros((2, nstate))   # r2 is 2D
        self.zi_r3 = np.zeros((2, nstate))   # r3 is 2D

    def updateResiduals(self, y1, y2, y3, y4,dt):
        """
        Update residuals
        """
        self.y1 = np.array(y1) / self.ppm
        self.y2 = y2
        self.y3 = y3
        self.y4 = y4

       

        if(self.initialize == 1):

            psi_k   = float(self.y3[0])
            psi_km1 = float(self.y3_previous)
            psi_mid = psi_k# wrap_pi(psi_km1 + 0.5*wrap_pi(psi_k - psi_km1))

            #Generate residual 1
            dy3 = wrap_pi(self.y3[0] - self.y3_previous) / dt
            r1 = (dy3 - self.y2[2] )
            #Filter and update filter values
            r1_f, self.zi_r1 = signal.lfilter(self.b, self.a, [r1], zi=self.zi_r1)
            self.r1.append(r1_f[0])


            #Generate residual 2
            xdot = ((self.y1[0] - self.y1_previous[0]) / dt)
            ydot = ((self.y1[1] - self.y1_previous[1]) / dt)
            psidot = self.y2[2]

            eta_dot_fd = np.array([xdot, ydot], dtype=float) 

            eta_calc = np.array([y2[0] * math.cos(self.y3[0]) - y2[1]* math.sin(self.y3[0]), y2[0]*math.sin(self.y3[0]) + y2[1] * math.cos(self.y3[0]) ])
            r2 = eta_dot_fd - eta_calc #( R(psi_mid) @ np.asarray(self.y2, dtype=float) )

            r2_f = np.zeros(2)
            for i in range(2):
                yi, self.zi_r2[i] = signal.lfilter(self.b, self.a, [r2[i]], zi=self.zi_r2[i])
                r2_f[i] = yi[0]
            self.r2.append(r2_f)
            #Generate residual 3
            
            zi_global = point_matching.pointCloud_screen_global(y4,[y1[0],y1[1],y3[0]])
            associations = point_matching.match_points([(self.lx,self.ly)],zi_global,15,self.radius,self.ppm)
            l_meassured = point_matching.meassured_circle(associations,self.radius)
            r3 =([self.lx,self.ly] - l_meassured) / self.ppm
            r3_f = np.zeros(2)
            for i in range(2):
                yi, self.zi_r3[i] = signal.lfilter(self.b, self.a, [r3[i]], zi=self.zi_r3[i])
                r3_f[i] = yi[0]
            self.r3.append(r3_f)
            #Update previous measurements
        
        self.y1_previous = self.y1
        self.y2_previous = self.y2
        self.y3_previous = self.y3[0]
        self.initialize = 1




    def getResiduals(self):
        """
        Return residuals
        """
        return self.r1,self.r2,self.r3