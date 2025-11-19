import numpy as np
from scipy.spatial import distance
from scipy.optimize import least_squares
import math

def match_points(landmarks,data_points,thershold,radii,ppm):
    associations = []

    for i, p in enumerate(data_points):
        #print(p)
        dists = distance.cdist([p], landmarks)[0] #Euclidian distance
        dists = dists - np.array(radii) #Subtract radius to match surface distances
        dists = dists / ppm             #Normalize based on pixel distances
        j = np.argmin(dists)
        if dists[j] < thershold:
            associations.append((j,p))  # (point_index, landmark_index, distance)
        else:
            #associations.append((i, None, None,None))  # no match
            pass
           
    #Prints Mismatch between meassured points and landmark pose.
    #for (i, j, d) in associations:
    #    if j is not None:
    #        print(f"Point {i} matched to Landmark {j} (distance = {d:.2f} m)")
    #    else:
    #        pass
            #print(f"Point {i} has no valid match (too far)")
    return associations

def pointCloud_screen_global(sensor_data):
    pts = []
    #d_px is a Distance in pixels
    #a_screen is the angle of the lidar scan it is in screen coordinates
    for d_px, a_screen, shipPose in sensor_data:
        sx, sy = shipPose
        xg =  d_px*np.cos(a_screen) + sx
        yg = -d_px*np.sin(a_screen) + sy  # screen y-down
        pts.append((xg, yg))
    return np.asarray(pts, float) #Returns x,y point cloud

def expected_distance(pose,landmark,radius,ppm):
    #Return expected distance between vessel and landmark
    #Euclidain Distance
    px=(pose[0]-landmark[0])**2
    py=(pose[1]-landmark[1])**2
    d = (math.sqrt(px+py))/ppm
    #print("Expected Distance Math: ",d)
    return d

def wrap_pi(angle):
    return (angle + math.pi) % (2*math.pi) - math.pi

def expected_bearing(pose,landmark):
    #Return expected distance between vessel and landmark
    #Euclidain Distance
    dy = -(landmark[1] -  pose[1])
    dx = (landmark[0] - pose[0])
    beta = math.atan2(dy,dx)
    theta = wrap_pi(beta - pose[2])
    psi = wrap_pi(pose[2])
   # print("Psi: ", math.degrees(psi))
   # print("Global Heading Beta: ",math.degrees(beta))
   
    #print("Theta: ",math.degrees(theta))
    
    return theta

def meassured_bearing(pose,c_hat):
    dy = -(c_hat[1] -  pose[1])
    dx = (c_hat[0] - pose[0])

    beta = math.atan2(dy,dx)
    theta = wrap_pi(beta - pose[2])
    #print("Theta Meassured: ", math.degrees(theta))

    return theta

def meassured_circle(associations,radius):
    #Return expected distance between vessel and landmark
    points = np.array([a[1] for a in associations if a[1] is not None])
    c0 = points.mean(axis=0)

    def residuals(c):
        return np.linalg.norm(points - c, axis=1) - radius
    #print(points.size)
    result = least_squares(residuals, c0)
    center_hat = result.x
    #residuals_final = residuals(center_hat)
    return center_hat


def meassured_distance(pose,c_hat,ppm):
    b = pose[1] - c_hat[1]
    a = pose[0] - c_hat[0]

    d = math.sqrt(a**2 + b**2)
    d = d/ppm
    return d