import pygame
import math
import numpy as np

def addUncertainty( distance, angle, sigma):
    mean = np.array([distance, angle])
    covar = np.diag(sigma**2)
    distance, angle = np.random.multivariate_normal(mean,covar)
    distance = max(distance,0)
    angle = max(angle,0)
    return [distance,angle]


class LaserSensor:
    def __init__(self,resolution,range,map,uncertainty):
        self.range = range
        self.resolution = resolution
        self.speed = 4 #Rotations per second
        self.sigma = np.array([uncertainty[0],uncertainty[1]])
        self.position=(0,0)
        self.map = map
        self.W, self.H =  map.get_width(), map.get_height()
        self.sensedObstacles=[]
    #Calculate a euclidian distance between own position and obstacle position
    def distance(self,obstaclePosition): 
        px=(obstaclePosition[0]-self.position[0])**2
        py=(obstaclePosition[1]-self.position[1])**2
        return math.sqrt(px+py)
    def senseObstacle(self):
        data=[]
        #Own position
        x1, y1 = self.position[0], self.position[1]
        #Create a list of angles between 0 and 2pi, 60 values
        for angle in np.linspace(0,2*math.pi, self.resolution, False):
            #x2 and y2 are the end points of the line limited by the range of the sensor.
            x2,y2 = (x1 + self.range * math.cos(angle), y1 - self.range * math.sin(angle))
            #From each angle point check a set distance untill we reach an object, from 0 to 100
            for i in range(0,100):
                u = i/100 #u is scaling the line from 0 to 1

                #Interpolation between own position and range end position
                x = int(x2 * u + x1 * ( 1 - u ))
                y = int(y2 * u + y1 * ( 1 - u ))
                if 0<x<self.W and 0<y<self.H: #Check if x and y are in the frame
                    color=self.map.get_at((x,y)) #Get color of position on line
                    if (color[0],color[1],color[2]) == (0,0,0): #Check if color of pixel is black
                        distance = self.distance((x,y)) #If object found set distance between found object pose and own pose
                        output = addUncertainty(distance,angle,self.sigma)
                        output.append(self.position)
                        data.append(output)
                        break #Break interpolation of line
        if len(data)>0:
            return data
        else:
            return False
    def pointCloud(self,data,psi): #In robot frame
        pointCloud = []
        for d,a_screen, _ in data:
            beta = -a_screen - psi                 # relative bearing in robot frame
            x = d * math.cos(beta)
            y = d * math.sin(beta)
            pointCloud.append((x,y))
        return np.array(pointCloud)
        

    