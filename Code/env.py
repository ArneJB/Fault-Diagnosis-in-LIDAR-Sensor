import pygame
import math

class buildEnv:
    def __init__(self,MapDimensions):
        pygame.init()
        self.pointCloud = []

        # --- FULL-RES MAP (from file) ---
        self.externalMap = pygame.image.load("map.png")
        self.OriginalMap = self.externalMap.copy()
        self.infomap     = self.OriginalMap.copy()    # used for drawing LiDAR points
        self.map         = self.OriginalMap.copy()    # used for drawing ship, circles, etc.

        # --- SMALL DISPLAY WINDOW ---
        self.win_h, self.win_w = MapDimensions
        pygame.display.set_caption("Lidar Test Map")
        self.window = pygame.display.set_mode((self.win_w, self.win_h))  # what you SEE

    #
    def AD2Pos(self,distance,angle,shipPose):
        x =  distance * math.cos(angle) + shipPose[0]
        y = -distance * math.sin(angle) + shipPose[1]
        return (int(x),int(y))
    
    def dataStorage(self,data):
        #print(len(self.pointCloud)) #Print number of points registered
        self.pointCloud = [] #Reset point cloud
        for element in data:
            point=self.AD2Pos(element[0],element[1],element[2])
            if point not in self.pointCloud:
                self.pointCloud.append(point)
    
    def show_sensorData(self):
        self.infomap=self.OriginalMap.copy()
        for point in self.pointCloud:
            pygame.draw.circle(self.infomap,(255,0,0),center=((int(point[0])),(int(point[1]))),radius=6)
   
    def render(self):
        """
        Scale the full-resolution self.map to the small window.
        Call this once per frame after you've drawn ship, circles, etc. on self.map.
        """
        fullres = self.infomap.copy()
        fullres.blit(self.map, (0, 0))  # ship / circles drawn in main loop go on self.map

        # Scale for window:
        scaled = pygame.transform.smoothscale(fullres, (self.win_w, self.win_h))
        self.window.blit(scaled, (0, 0))
        pygame.display.update()