import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import pygame
import control as ctrl
#Own Functions:
import vessel_kinematics
import env
import sensor
import reference_detector
import point_matching
import glr
import gps
import gyroCompass
import imu

import structural

import poseObserver_unkownVelocities
#import velObserver
import velObserver_unkownModelVelocities
import imu_gnss_Observer
#import Observer_Lidar_GNSS
import Observer_Lidar_unkownVelocities
import Observer5_LidarGNSS
# Rotation matrix R(psi)
# --------------------------------------------------------
def R(psi):
    return np.array([
        [np.cos(psi), -np.sin(psi), 0],
        [np.sin(psi),  np.cos(psi), 0],
        [0,            0,           1]
    ])

def wrap_pi(angle):
    """Wrap angle to [-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


# ------------------------------
# Ship initial pose in pixels
# ------------------------------
x = 2000
y = 700
psi = 0.0



# Body velocities (Fossen: u=surge, v=sway)
u = 1.5                   # forward m/s
v = 0.0                   # Sideway m/s
r = 10 * math.pi/180      # Yaw rate angular velocity rad/s
tau_u, tau_v, tau_r = 20.0, 20.0, 20.0 #SMOOTHING OF VELOCITIES
u_cmd, v_cmd, r_cmd = u, v, r

ppm = 50 #15 pixels equates to 1 meter. map will be 40m x 80m

# 0 UNCERTAINTY CONDITION 
#gps_sensor = gps.GPS([x,y],[0*ppm,0*ppm]) #Create GPS sensor 1 meter std. in x and y
#gyro_compas = gyroCompass.gyroCompass(psi,0 * math.pi/180) #Create Gyro Compas sensor 0.5 degree std.

gps_sensor = gps.GPS([x,y],[1*ppm,1*ppm]) #Create GPS sensor 1 meter std. in x and y
gyro_compas = gyroCompass.gyroCompass(psi,0.5 * math.pi/180) #Create Gyro Compas sensor 0.5 degree std.

#poseObs = poseObserver.PoseObserver(x,y,psi,ppm,[0.1,0.1,0.1]) #Create Observer with known velocities from body
poseObs = poseObserver_unkownVelocities.PoseVelObserver(x,y,psi,ppm) #Create Observer without know velocities

# 0 UNCERTAINTY CONDITION 
#sigma_u = 0.0      
#sigma_v = 0.0
#sigma_r = 0.0

sigma_u = 0.02      
sigma_v = 0.02
sigma_r = 0.2 * math.pi/180  # 0.2 deg/s
imu_sensor = imu.IMU([sigma_u, sigma_v, sigma_r])

# 0 UNCERTAINTY CONDITION 
#velObs = velObserver.VelObserver(u,v,r,psi,0.0,0.0,0.0,0.0)
velObs = velObserver_unkownModelVelocities.VelObserver(u,v,r,psi,0.6,0.6,0.6,0.9)

# 0 UNCERTAINTY CONDITION 
#imuGnssObs = imu_gnss_Observer.gnssImuObserver(x,y,psi,ppm,u,v,[0.0,0.0,0.000])
imuGnssObs = imu_gnss_Observer.gnssImuObserver(x,y,psi,ppm,u,v,[0.1,0.1,0.00,0.4,0.4,0.4])

#Create Vessel object
ship = vessel_kinematics.marineVessel(x,y,psi)
ship.setVelocities(u,v,r)

enviorment = env.buildEnv((800,1400)) #1200,600 pixel map.

#Sensor resolution and range
resolution = 500   #Number of points in a 360 degree scan
Sensor_range = 15 * ppm   #50 meters * pixels per meter.
print("Map Size(m): ", [3810/ppm, 2096/ppm])

# 0 UNCERTAINTY CONDITION 
#laser=sensor.LaserSensor(resolution,Sensor_range,enviorment.OriginalMap,uncertainty=(0.0,0.0))
laser=sensor.LaserSensor(resolution,Sensor_range,enviorment.OriginalMap,uncertainty=(0.02,0.005))

enviorment.map.fill((0,0,0))
enviorment.infomap = enviorment.map.copy()
#Reference circle location
cx, cy, radius = reference_detector.detect_circle_opencv("Map.png")
#  = lm[2][0], lm[2][1], lm[2][2]
print("Center: x, y, radius in meters:", cx, cy, radius/ppm) 
print("Reference size m^2: ", math.pi*(radius/ppm)**2)
#11 Meter Radius Landmark


# 0 UNCERTAINTY CONDITION 
#lidar_gnssObs = Observer_Lidar_GNSS.lidar_gnss_observer(x, y, psi, ppm, u, v, [0.0,0.0])
#lidar_gnssObs = Observer_Lidar_GNSS.lidar_gnss_observer(x, y, psi, ppm, u, v, [0.1,0.1])
lidar_gyroObs = Observer_Lidar_unkownVelocities.lidar_gyro_observer(x, y, psi, ppm, [0.05,0.05,0.1,0.4,0.99],[cx,cy],radius)

lidar_gnssObs = Observer5_LidarGNSS.lidar_gnss_observer(x, y,ppm,[0.4,0.4,0.6,0.2],[cx,cy],radius)

structAnalysis = structural.structuralResidual(0, 0, 0,0,[cx,cy],radius,ppm)

#Ship length
L = 2 * ppm #2 Meter long vessel
print("Vessel length in meters: ", L/ppm)
print("LIDAR map:", laser.map.get_size())
running = True

time_scale = 20

dt = 1/60
clock = pygame.time.Clock()
font = pygame.font.Font(None, 60) 
t_ms = 0

sim_time = np.array([0.0]) #Track time passed in simulation.
r1 = np.array([])
r2 = np.array([])

r_poseObs = []
r_velObs = []
r_imuGnss = []
r_lidargyro = []

psi_meas_arr = []
psi_hat_1_arr = []

u_imposed = []
v_imposed = []
r_imposed = []

u_hat1, v_hat1, r_hat1 = 0,0,0
x_true,y_true,psi_true = [],[],[]
x_est,y_est, psi_est = [],[],[]
u_est = []
v_est = []
r_est = []

d_true, d_est = [],[]

u_est2,v_est2,r_est2, psi_est2 = [],[],[],[]
x_est3,y_est3, r_est3 = [],[],[]
x_est4,y_est4, psi_est4 = [],[],[]

x_est5,y_est5,d_est5 = [],[],[]
r_lidargnss = []

fault_gyrocompas = 0
fault_compas_bias = 0
fault_imu_var = [0, 0, 0]
fault_imu_bias = [0, 0, 0]

theta_arr = np.array([])
theta_meas_arr = np.array([]) 
n = 0 #Itterator 
while running:
    sensorOn = False
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    x,y,psi = ship.updatePose(dt,ppm)
    x_true.append(x)
    y_true.append(y)
    psi_true.append(wrap_pi(psi))
    #While psi is calculated as a global frame pygame y is flipped.
    psi_real = -psi

    x_measured, y_measured, course_over_ground = gps_sensor.gpsPose(dt,[x,y])
    psi_meassured = wrap_pi(gyro_compas.psi_measured(psi,fault_compas_bias,fault_gyrocompas))
    psi_meas_arr.append(psi_meassured)

    u_meas, v_meas, r_meas = imu_sensor.measure([u_cmd,v_cmd,r_cmd],fault_imu_var,fault_imu_bias)

    if n <= 0:
        x_hat1 = x_measured
        y_hat1 = y_measured
        psi_hat1 = psi_meassured
        poseObs.x_hat = x_hat1
        poseObs.y_hat = y_hat1
        poseObs.psi_hat = psi_hat1
    else:
        #x_hat1, y_hat1, psi_hat1 = poseObs.step(dt, u,v,r, x_measured,y_measured,psi_meassured)
        x_hat1, y_hat1, psi_hat1, u_hat1, v_hat1, r_hat1 = poseObs.step(dt, x_measured,y_measured,psi_meassured)
    x_est.append(x_hat1), y_est.append(y_hat1), psi_est.append(psi_hat1)
    u_est.append(u_hat1)
    v_est.append(v_hat1)
    r_est.append(r_hat1)
    epsilon_1 = [(x_measured - x_hat1)/ppm, (y_measured - y_hat1)/ppm, wrap_pi(psi_meassured - psi_hat1)]
    r_poseObs.append(epsilon_1)
    psi_hat_real = -psi_hat1
    psi_hat_1_arr.append(psi_hat1)
    #Velocity based observer
    u_hat, v_hat, r_hat, psi_hat2 = velObs.step(dt, u_meas, v_meas, r_meas,  psi_meassured)
    #Save observer estimates
    u_est2.append(u_hat)
    v_est2.append(v_hat)
    r_est2.append(r_hat)
    psi_est2.append(psi_hat2)
    
    epsilon_2 = [(u_meas - u_hat),(v_meas-v_hat),(r_meas-r_hat), wrap_pi(psi_meassured - psi_hat2)]
    r_velObs.append(epsilon_2)


    #IMU GNSS OBSERVER
    x_hat3,y_hat3,psi3, r_hat3 = imuGnssObs.step(dt,u_meas,v_meas,r_meas,x_measured,y_measured,course_over_ground)
    x_est3.append(x_hat3)
    y_est3.append(y_hat3)
    r_est3.append(r_hat3)
    epsilon_3 = [(x_measured-x_hat3)/ppm,(y_measured-y_hat3)/ppm, wrap_pi(r_meas-r_hat3)]
    r_imuGnss.append(epsilon_3)


    # --- Update LiDAR position with true pose --- #
    laser.position = (int(x), int(y))
    #print(laser.position)
    fault_distance = True if sim_time[n] > 300 else False
    fault_angle    = True if sim_time[n] > 300 else False
    #laser.sigma = np.array([10,0.2]) if sim_time[n] > 30 else np.array([0.02,0.005])

    sensor_data = laser.senseObstacle(psi,fault_distance,fault_angle) #Raw sensor data
    #Save to global frame for plotting in map.
    if sensor_data:
        enviorment.dataStorage(sensor_data,psi)
    enviorment.show_sensorData()
    
    #Create structural analysis residuals from meassurements
    structAnalysis.updateResiduals([x_measured, y_measured],[u_meas, v_meas, r_meas],[psi_meassured],sensor_data,dt)

    #LIDAR GNSS OBSERVER
    #x_4, y_4, psi_4 = lidar_gnssObs.step(dt, u_cmd, v_cmd, r_cmd, x_measured, y_measured)
    x_4, y_4, psi_4, d_4 = lidar_gyroObs.step(dt,sensor_data,psi_meassured)
    x_est4.append(x_4)
    y_est4.append(y_4)
    psi_est4.append(psi_4)
    d_est.append(d_4)

    x_5,y_5,d_5 = lidar_gnssObs.step(dt,sensor_data,x_measured,y_measured)
    x_est5.append(x_5)
    y_est5.append(y_5)
    d_est5.append(d_5)

    #Take point cloud and associate the meassrued points to the known landmark. Done by taking matching point positions to landmark position
    pointCloudGlobal = point_matching.pointCloud_screen_global(sensor_data,[x_4,y_4,psi_4])
    associations = point_matching.match_points([(cx,cy)],pointCloudGlobal,3,radius,ppm)
    #RAW LIDAR MEASUREMENT 
    pointCloudBody = point_matching.pointCloud_body(sensor_data)
    associations_raw = point_matching.match_points_raw(pointCloudBody)
    c_hat_raw = point_matching.meassured_circle(associations_raw,radius)
    meassured_dist  = point_matching.meassured_distance((0,0),c_hat_raw,ppm)

    #True Distance to landmark
    dist_to_landmark = point_matching.expected_distance((x,y),(cx,cy),radius,ppm)
    d_true.append(dist_to_landmark)

    epsilon_4 = [(meassured_dist-d_4), wrap_pi(psi_meassured-psi_4)]
    r_lidargyro.append(epsilon_4)

    epsilon_5 = [(x_measured-x_5),(y_measured - y_5 ), (meassured_dist-d_5)]
    r_lidargnss.append(epsilon_5)

    for (xg, yg) in pointCloudGlobal:
        pygame.draw.circle(enviorment.infomap, (0,0,255), (int(xg), int(yg)), 5)
    
    enviorment.map.blit(enviorment.infomap, (0,0))
    #Meassured
    if not associations:
        print("No Points Associated to Land Mark")
    else:
        c_hat = point_matching.meassured_circle(associations,radius)
        #meassured_dist  = point_matching.meassured_distance((x_4,y_4),c_hat,ppm)
        #meassured_theta = point_matching.meassured_bearing((x_4,y_4,psi_4),c_hat)
        #theta_meas_arr = np.append(theta_meas_arr,meassured_theta)
        #Model

        #theta = point_matching.expected_bearing((x_4,y_4,psi_4),(cx,cy)) #We pass the inverted heading to match the visual expectation of the sim
        #theta_arr = np.append(theta_arr,theta)

        #r1 = np.append(r1, (meassured_dist - dist_to_landmark))
        #Theta represents an angel make sure no wrap around spikes occur when comparing them, theta meassure and model theta are 2pi and 0
        #r2 = np.append(r2, point_matching.wrap_pi(meassured_theta - theta))         #print("r1 (in m): ", r1 )
        #print("r2: ", r2)
        #print("Distance: ", dist_to_landmark)
    bow_x = x + L * np.cos(psi)
    bow_y = y + L * np.sin(psi)

    bow_x_hat = x_hat1 + L * np.cos(psi_hat1)
    bow_y_hat = y_hat1 + L * np.sin(psi_hat1)

    bow_x_hat3 = x_hat3 + L * np.cos(psi3)
    bow_y_hat3 = y_hat3 + L * np.sin(psi3)

    bow_x4 = x_4 + L * np.cos(psi_4)
    bow_y4 = y_4 + L * np.sin(psi_4)
    #Draw ship
    pygame.draw.line(enviorment.map, (0,255,0), (x,y), (bow_x,bow_y), 5)
    #GNSS Gyrocompass
    pygame.draw.line(enviorment.map, (0,255,255), (x_hat1,y_hat1), (bow_x_hat,bow_y_hat), 5)
    #IMU GNSS
    pygame.draw.line(enviorment.map, (255,255,0), (x_hat3,y_hat3), (bow_x_hat3,bow_y_hat3), 5)

    pygame.draw.line(enviorment.map, (0,0,0), (x_4,y_4), (bow_x4,bow_y4), 5)

    #Draw Estimated Circle
    pygame.draw.circle(enviorment.map, (255,0,0), (int(c_hat[0]), int(c_hat[1])), radius, 4)
    #Draw true reference object
    pygame.draw.circle(enviorment.map, (0,255,255), (cx, cy), radius, 4)
    
    # --- Draw simulation time ---  
    time_text = font.render(f"Time: {sim_time[n]:.1f} s", True, (0, 0, 0))

    # Position in top-right corner
    text_rect = time_text.get_rect()
    text_rect.topright = (enviorment.map.get_width() - 20, 20)

    enviorment.map.blit(time_text, text_rect)

    enviorment.render()

    clock.tick(60)
    t_ms = clock.get_time() 
    
    #print(pygame.time.get_ticks()/1000)
    #dt = t_ms/1000.0 #Update sampling time it is not fixed due to the nature of pygame
    dt = 1/60
    dt = dt * time_scale

    sim_time = np.append(sim_time, sim_time[n] + dt) #Time Axis
    if(sim_time[n] < 30):
        ship.setVelocities(u,v,r)
    elif(sim_time[n] < 60):
        #fault_gyrocompas = 50 * math.pi/180 #Variance increase
        #fault_compas_bias = 25 * math.pi/180
        u = -1.5                   # forward m/s
        v = -0.0                   # Sideway m/s
        r = -10 * math.pi/180      # Yaw rate angular velocity rad/s
    elif(sim_time[n] < 65): 
        u = -1.0
        v=0.0
        r=0.0
    elif(sim_time[n] < 75):
         fault_imu_bias = [0.6, 0.6, 0.2]
         #fault_imu_var = [1.3, 1.3, 1.1]         
         u=0.0
         v=1.2
         r=0.0
    elif(sim_time[n] < 90):
         u = 1.0
         v = 0.0
         r = 0.0    
    elif(sim_time[n] < 110):
        #gps_sensor.setUncertainty([3.5*ppm,3.5*ppm],0)
        #gps_sensor.setUncertainty([1*ppm,1*ppm],400) #FOR OBSERVER

        #fault_imu_bias = [0, 0, 0]
        u = 0.0
        v = -0.4
        r = 0.0
    elif(sim_time[n] < 160):
        u = 0.0
        v = 0.4
        r = 10 * math.pi/180
    elif(sim_time[n] < 170):
        #fault_imu_var = [0, 0, 1]
        #fault_imu_bias = [0.5, 0.5, 0.5]
        u = 0.1
        v = 0.0
        r = 0 * math.pi/180

    alpha_u = 1.0 - math.exp(-dt / tau_u)
    alpha_v = 1.0 - math.exp(-dt / tau_v)
    alpha_r = 1.0 - math.exp(-dt / tau_r)

    u_cmd += alpha_u * (u - u_cmd)
    v_cmd += alpha_v * (v - v_cmd)
    r_cmd += alpha_r * (r - r_cmd)
    ship.setVelocities(u_cmd,v_cmd,r_cmd )

    if(sim_time[n] > 170):
        running = False
    #print("Simulatedtime:", sim_time)
    u_imposed.append(u_cmd)
    v_imposed.append(v_cmd)
    r_imposed.append(r_cmd)
    n+=1

v_imposed = np.array(v_imposed)
u_imposed = np.array(u_imposed)
r_imposed = np.array(r_imposed)

plt.figure()
plt.plot(sim_time[0:-1],u_imposed,label="u")
plt.plot(sim_time[0:-1],v_imposed,label="v")
plt.plot(sim_time[0:-1],r_imposed,label="r")
plt.xlabel("Time[s]")
plt.ylabel("Velocities")
plt.legend()
plt.grid(True)
plt.title("Imposed Velocities")
plt.tight_layout()
plt.show()

# ##################################    Structural Analysis    ##############################################
if(1):
    print("Structural Analysis Residuals")
    [r1_SA,r2_SA,r3_SA] = structAnalysis.getResiduals()
    r1_SA = np.array(r1_SA)
    r2_SA = np.array(r2_SA)
    r3_SA = np.array(r3_SA)

   # ================= FIGURE LAYOUT =================
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.3
    )

    ax_r1   = fig.add_subplot(gs[:, 0])   # span both rows
    ax_r2_1 = fig.add_subplot(gs[0, 1])
    ax_r2_2 = fig.add_subplot(gs[1, 1])
    ax_r3_1 = fig.add_subplot(gs[0, 2])
    ax_r3_2 = fig.add_subplot(gs[1, 2])

    # ================= R1 =================
    ax_r1.plot(sim_time[0:-2], r1_SA, label=r"$r_\psi$")
    ax_r1.set_title("Structural Analysis Residual r1")
    ax_r1.set_xlabel("Time [s]")
    ax_r1.set_ylabel("Residual")
    ax_r1.grid(True)
    ax_r1.legend()

    print("Mean R1: ", np.mean(r1_SA))
    print("Var  R1: ", np.var(r1_SA))

    # ================= R2 =================
    ax_r2_1.plot(sim_time[0:-2], r2_SA[:, 0], label=r"$r_{\dot{x}}$")
    ax_r2_1.set_title("Structural Analysis Residual r2[1]")
    ax_r2_1.grid(True)
    ax_r2_1.legend()

    ax_r2_2.plot(sim_time[0:-2], r2_SA[:, 1], label=r"$r_{\dot{y}}$")
    ax_r2_2.set_title("Structural Analysis Residual r2[2]")
    ax_r2_2.set_xlabel("Time [s]")
    ax_r2_2.grid(True)
    ax_r2_2.legend()

    print("Mean R2_1: ", np.mean(r2_SA[:, 0]))
    print("Var  R2_1: ", np.var(r2_SA[:, 0]))
    print("Mean R2_2: ", np.mean(r2_SA[:, 1]))
    print("Var  R2_2: ", np.var(r2_SA[:, 1]))

    # ================= R3 =================
    ax_r3_1.plot(sim_time[0:-2], r3_SA[:, 0], label=r"$r_{l_x}$")
    ax_r3_1.set_title("Structural Analysis Residual r3[1]")
    ax_r3_1.grid(True)
    ax_r3_1.legend()

    ax_r3_2.plot(sim_time[0:-2], r3_SA[:, 1], label=r"$r_{l_y}$")
    ax_r3_2.set_title("Structural Analysis Residual r3[2]")
    ax_r3_2.set_xlabel("Time [s]")
    ax_r3_2.grid(True)
    ax_r3_2.legend()

    print("Mean R3_1: ", np.mean(r3_SA[:, 0]))
    print("Var  R3_1: ", np.var(r3_SA[:, 0]))
    print("Mean R3_2: ", np.mean(r3_SA[:, 1]))
    print("Var  R3_2: ", np.var(r3_SA[:, 1]))

    plt.show()


    #############  GLR   #############

    glr_structural_r1 = glr.GLR(WindowSize=30,residual=r1_SA,mean=0.0,sigma = 5e-3)
    g_struct_r1, idx_r1, mu_1_r1 = glr_structural_r1.computeGlr()


    glr_structural_r2_1 = glr.GLR(WindowSize=10,residual=r2_SA[:,0],mean=0.0,sigma = 0.3)
    g_struct_r2_1, idx_r2_1, mu_1_r2_1 = glr_structural_r2_1.computeGlr()

    glr_structural_r2_2 = glr.GLR(WindowSize=10,residual=r2_SA[:,1],mean=0.0,sigma = 0.3)
    g_struct_r2_2, idx_r2_2, mu_1_r2_2 = glr_structural_r2_2.computeGlr()

    glr_structural_r3_1 = glr.GLR(WindowSize=10,residual=r3_SA[:,0],mean=5,sigma = 5)
    g_struct_r3_1, idx_r3_1, mu_1_r2_1 = glr_structural_r3_1.computeGlr()

    glr_structural_r3_2 = glr.GLR(WindowSize=10,residual=r3_SA[:,1],mean=5,sigma = 5)
    g_struct_r3_2, idx_r3_2, mu_1_r2_2 = glr_structural_r3_2.computeGlr()

    fig = plt.figure()
    gs = GridSpec(
        nrows=2,
        ncols=3,
        width_ratios=[1, 1, 1],
        height_ratios=[1, 1],
        hspace=0.3,
        wspace=0.3
    )

    # ================= AXES =================
    ax_r1   = fig.add_subplot(gs[:, 0])    # spans both rows
    ax_r2_1 = fig.add_subplot(gs[0, 1])
    ax_r2_2 = fig.add_subplot(gs[1, 1])
    ax_r3_1 = fig.add_subplot(gs[0, 2])
    ax_r3_2 = fig.add_subplot(gs[1, 2])

    # ================= PLOT r1 =================
    ax_r1.plot(sim_time[0:-2], g_struct_r1, label="GLR g(k)")
    ax_r1.axhline(8, color='red', linestyle='--', label='Threshold h')
    ax_r1.set_title("GLR - r1")
    ax_r1.set_xlabel("Time [s]")
    ax_r1.grid(True)
    ax_r1.legend()

    # ================= PLOT r2 =================
    ax_r2_1.plot(sim_time[0:-2], g_struct_r2_1, label="GLR g(k)")
    ax_r2_1.axhline(8, color='red', linestyle='--')
    ax_r2_1.set_title("GLR - r2[1]")
    ax_r2_1.grid(True)

    ax_r2_2.plot(sim_time[0:-2], g_struct_r2_2, label="GLR g(k)")
    ax_r2_2.axhline(8, color='red', linestyle='--')
    ax_r2_2.set_title("GLR - r2[2]")
    ax_r2_2.set_xlabel("Time [s]")
    ax_r2_2.grid(True)

    # ================= PLOT r3 =================
    ax_r3_1.plot(sim_time[0:-2], g_struct_r3_1, label="GLR g(k)")
    ax_r3_1.axhline(8, color='red', linestyle='--')
    ax_r3_1.set_title("GLR - r3[1]")
    ax_r3_1.grid(True)

    ax_r3_2.plot(sim_time[0:-2], g_struct_r3_2, label="GLR g(k)")
    ax_r3_2.axhline(8, color='red', linestyle='--')
    ax_r3_2.set_title("GLR - r3[2]")
    ax_r3_2.set_xlabel("Time [s]")
    ax_r3_2.grid(True)
    fig.suptitle("GLR Structural Analysis")
    plt.show()


#########################################################################
############### ------ Observer 1 - GNSS + GYROCOMPASS ------ ###############
if(1): #PLOT ESTIMATED VS TRUE VELOCITIES OBSERVER 1
    print("Samples: ", n)
    x_true = np.array(x_true)
    y_true = np.array(y_true)
    psi_true = np.array(psi_true)
    u_est = np.array(u_est)
    v_est = np.array(v_est)
    r_est = np.array(r_est)
    x_est = np.array(x_est)
    y_est = np.array(y_est)
    psi_est = np.array(psi_est)

    fig, axs = plt.subplots(3, 1)

    axs[0].plot(sim_time[:-1], u_imposed, 'k--', label='True u')
    axs[0].plot(sim_time[:-1], u_est, 'r-', label='Estimated u')
    axs[0].set_ylabel('u')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], v_imposed, 'k--', label='True v')
    axs[1].plot(sim_time[:-1], v_est, 'b-', label='Estimated v')
    axs[1].set_ylabel('v')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sim_time[:-1], r_imposed, 'k--', label='True r')
    axs[2].plot(sim_time[:-1], r_est, 'g-', label='Estimated r')
    axs[2].set_ylabel('r')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("Estimated Velocities - Observer 1")
    fig.tight_layout()
    
    fig, axs = plt.subplots(3, 1)

    axs[0].plot(sim_time[:-1], x_true, 'k--', label='True x')
    axs[0].plot(sim_time[:-1], x_est, 'r-', label='Estimated x')
    axs[0].set_ylabel('u')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], y_true, 'k--', label='True y')
    axs[1].plot(sim_time[:-1], y_est, 'b-', label='Estimated y')
    axs[1].set_ylabel('v')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sim_time[:-1], psi_true, 'k--', label='True psi')
    axs[2].plot(sim_time[:-1], psi_est, 'g-', label='Estimated psi')
    axs[2].set_ylabel('r')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("Estimated Pose - Observer 1")
    fig.tight_layout()
    plt.show()


if(1): #PLOT GLR OBSERVER 1
    r_poseObs = np.array(r_poseObs)
    
    print("Observer 1: ")
    mean_x = np.mean(r_poseObs[:,0])
    std_x  = np.std(r_poseObs[:,0])
    
    mean_y = np.mean(r_poseObs[:,1])
    std_y  = np.std(r_poseObs[:,1])

    mean_psi = np.mean(r_poseObs[:,2])
    std_psi  = np.std(r_poseObs[:,2])

    print(f"x (mean, std): {mean_x:.5f}, {std_x:.5f}")
    print(f"y (mean, std): {mean_y:.5f}, {std_y:.5f}")
    print(f"psi (mean, std): {mean_psi:.5f}, {std_psi:.5f}")


    fig, ax = plt.subplots(3, 1)
    # --- Plot 1: r1_x ---
    ax[0].plot(sim_time[0:-1], r_poseObs[:,0], label="X")
    #ax[0].set_title("Residual x")
    ax[0].legend()
    ax[0].grid(True)

    # --- Plot 2: r1_y ---
    ax[1].plot(sim_time[0:-1], r_poseObs[:,1], label="Y")
    #ax[1].set_title("Residual y")
    ax[1].legend()
    ax[1].grid(True)

    # --- Plot 3: r1_psi ---
    ax[2].plot(sim_time[0:-1], r_poseObs[:,2], label="Psi)")
    #ax[2].set_title("Residual Psi")
    ax[2].legend()
    ax[2].grid(True)
    fig.suptitle("Residuals - Observer 1")
    # Improve layout
    plt.xlabel("Time [s]")

    glr_r1_1 = glr.GLR(WindowSize=10,residual=r_poseObs[:,0],mean=0.0,sigma=1.2)
    g_r1, idx_r1, mu_1_r1 = glr_r1_1.computeGlr()

    glr_r1_2 = glr.GLR(WindowSize=10,residual=r_poseObs[:,1],mean=0.0,sigma=1.2)
    g_r2, idx_r2, mu_1_r2 = glr_r1_2.computeGlr()

    glr_r1_3 = glr.GLR(WindowSize=10,residual=r_poseObs[:,2],mean=0.0,sigma=0.001)
    g_r3, idx_r3, mu_1_r3 = glr_r1_3.computeGlr()

    fig, ax = plt.subplots(3, 1)
    # --- Plot 1: r1_x ---
    ax[0].plot(sim_time[0:-1], g_r1, label="GLR g(k)")
    ax[0].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[0].set_title("GLR Test Statistic for r₁ (x-direction)")
    ax[0].legend()
    ax[0].grid(True)

    # --- Plot 2: r1_y ---
    ax[1].plot(sim_time[0:-1], g_r2, label="GLR g(k)")
    ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[1].set_title("GLR Test Statistic for r₁ (y-direction)")
    ax[1].legend()
    ax[1].grid(True)

    # --- Plot 3: r1_psi ---
    ax[2].plot(sim_time[0:-1], g_r3, label="GLR g(k)")
    ax[2].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[2].set_title("GLR Test Statistic for r₁ (ψ-direction)")
    ax[2].legend()
    ax[2].grid(True)
    fig.suptitle("GLR - Observer 1")
    # Improve layout
    plt.xlabel("Time [s]")
        
    plt.tight_layout()
    plt.show()
    

############### ------ Observer 2 - GYROCOMPASS + GNSS ------ ###############
print("Observer 2: ")
if(1):
    u_hat, v_hat, r_hat, psi_hat2
    r_velObs = np.array(r_velObs)

    u_est2 = np.array(u_est2)
    v_est = np.array(v_est2)
    r_est = np.array(r_est2)
    psi_est2 = np.array(psi_est2)

    fig, axs = plt.subplots(4, 1)

    axs[0].plot(sim_time[:-1], u_imposed, 'k--', label='True u')
    axs[0].plot(sim_time[:-1], u_est2, 'r-', label='Estimated u')
    axs[0].set_ylabel('u')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], v_imposed, 'k--', label='True v')
    axs[1].plot(sim_time[:-1], v_est2, 'b-', label='Estimated v')
    axs[1].set_ylabel('v')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sim_time[:-1], r_imposed, 'k--', label='True r')
    axs[2].plot(sim_time[:-1], r_est2, 'g-', label='Estimated r')
    axs[2].set_ylabel('r')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    axs[3].plot(sim_time[:-1], psi_true, 'k--', label='True heading psi')
    axs[3].plot(sim_time[:-1], psi_est2, 'c-', label='Estimated psi')
    #axs[3].plot(sim_time[:-1], psi_meas_arr, 'r-', label='Meassured psi')
    axs[3].set_ylabel('psi')
    axs[3].set_xlabel('Time')
    axs[3].legend()
    axs[3].grid(True)

    fig.suptitle("Estimated Values - Observer 2")
    fig.tight_layout()

    mean_u = np.mean(r_velObs[:,0])
    std_u  = np.std(r_velObs[:,0])


    mean_v = np.mean(r_velObs[:,1])
    std_v  = np.std(r_velObs[:,1])

    mean_yaw = np.mean(r_velObs[:,2])
    std_yaw  = np.std(r_velObs[:,2])

    mean_psi = np.mean(r_velObs[:,3])
    std_psi  = np.std(r_velObs[:,3])

    print(f"u (mean, std): {mean_u:.5f}, {std_u:.5f}")
    print(f"v (mean, std): {mean_v:.5f}, {std_v:.5f}")
    print(f"yaw (mean, std): {mean_yaw:.5f}, {std_yaw:.5f}")
    print(f"psi (mean, std): {mean_psi:.5f}, {std_psi:.5f}")


    plt.figure()
    plt.plot(sim_time[0:-1],r_velObs[:,0],label="r_u")
    plt.plot(sim_time[0:-1],r_velObs[:,1],label="r_v")
    plt.plot(sim_time[0:-1],r_velObs[:,2],label="r_yaw")
    plt.plot(sim_time[0:-1],r_velObs[:,3],label="r_psi")
    plt.xlabel("Time[s]")
    plt.ylabel("residual")
    plt.legend()
    plt.grid(True)
    plt.title("Residuals - Observer 2")
    plt.tight_layout()
    plt.show()

    glr_r2_1 = glr.GLR(WindowSize=10,residual=r_velObs[:,0],mean=0.0,sigma=0.03)
    g_r1, idx_r1, mu_1_r1 = glr_r2_1.computeGlr()

    glr_r2_2 = glr.GLR(WindowSize=10,residual=r_velObs[:,1],mean=0.0,sigma=0.03)
    g_r2, idx_r2, mu_1_r2 = glr_r2_2.computeGlr()

    glr_r2_3 = glr.GLR(WindowSize=10,residual=r_velObs[:,2],mean=0.0,sigma=0.005)
    g_r3, idx_r3, mu_1_r3 = glr_r2_3.computeGlr()

    glr_r2_4 = glr.GLR(WindowSize=10,residual=r_velObs[:,3],mean=0.0,sigma=0.0005)
    g_r4, idx_r3, mu_1_r3 = glr_r2_4.computeGlr()


    fig, ax = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    # --- Plot 1: r2_u ---
    ax[0].plot(sim_time[0:-1], g_r1, label="GLR g(k)")
    ax[0].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[0].set_title("GLR Test Statistic for r2 u")
    ax[0].legend()
    ax[0].grid(True)

    # --- Plot 2: r2_v ---
    ax[1].plot(sim_time[0:-1], g_r2, label="GLR g(k)")
    ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[1].set_title("GLR Test Statistic for r2 v")
    ax[1].legend()
    ax[1].grid(True)

    # --- Plot 3: r2_yaw ---
    ax[2].plot(sim_time[0:-1], g_r3, label="GLR g(k)")
    ax[2].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[2].set_title("GLR Test Statistic for r2 yaw")
    ax[2].legend()
    ax[2].grid(True)

    # --- Plot 3: r2_psi ---
    ax[3].plot(sim_time[0:-1], g_r4, label="GLR g(k)")
    ax[3].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[3].set_title("GLR Test Statistic for r2 ψ")
    ax[3].legend()
    ax[3].grid(True)
    plt.xlabel("Time [s]")
    fig.suptitle("GLR - Observer 2")
    plt.show()




############### ------ Observer 3 - GNSS + IMU ------ ###############

print("Observer 3:")
if(1):
    r_imuGnss = np.array(r_imuGnss)

    x_est3 = np.array(x_est3)
    y_est3 = np.array(y_est3)
    r_est3 = np.array(r_est3)
    
    mean_x = np.mean(r_imuGnss[:,0])
    std_x  = np.std(r_imuGnss[:,0])

    mean_y = np.mean(r_imuGnss[:,1])
    std_y  = np.std(r_imuGnss[:,1])

    mean_psi = np.mean(r_imuGnss[:,2])
    std_psi  = np.std(r_imuGnss[:,2])

    print(f"x (mean, std): {mean_x:.5f}, {std_x:.5f}")
    print(f"y (mean, std): {mean_y:.5f}, {std_y:.5f}")
    print(f"y (mean, std): {mean_psi:.5f}, {std_psi:.3f}")



    fig, axs = plt.subplots(3, 1)

    axs[0].plot(sim_time[:-1], x_true, 'k--', label='True x')
    axs[0].plot(sim_time[:-1], x_est3, 'r-', label='Estimated x')
    axs[0].set_ylabel('x')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], y_true, 'k--', label='True y')
    axs[1].plot(sim_time[:-1], y_est3, 'b-', label='Estimated y')
    axs[1].set_ylabel('y')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sim_time[:-1], r_imposed, 'k--', label='True yaw rate')
    axs[2].plot(sim_time[:-1], r_est3, 'g-', label='Estimated yaw rate')
    axs[2].set_ylabel('psi')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("Estimated Values - Observer 3")
    fig.tight_layout()


    plt.figure()
    plt.plot(sim_time[0:-1],r_imuGnss[:,0],label="r_x")
    plt.plot(sim_time[0:-1],r_imuGnss[:,1],label="r_y")
    plt.plot(sim_time[0:-1],r_imuGnss[:,2],label="r_yaw")
    plt.xlabel("Time[s]")
    plt.ylabel("residual")
    plt.legend()
    plt.title("Residuals - Observer 3")
    plt.grid(True)
    plt.tight_layout()


    glr_r3_1 = glr.GLR(WindowSize=10,residual=r_imuGnss[:,0],mean=0.0,sigma=0.99)
    g_r1, idx_r1, mu_1_r1 = glr_r3_1.computeGlr()

    glr_r3_2 = glr.GLR(WindowSize=10,residual=r_imuGnss[:,1],mean=0.0,sigma=0.99)
    g_r2, idx_r2, mu_1_r2 = glr_r3_2.computeGlr()

    glr_r3_3 = glr.GLR(WindowSize=10,residual=r_imuGnss[:,2],mean=0.0,sigma=0.01)
    g_r3, idx_r3, mu_1_r3 = glr_r3_3.computeGlr()


    fig, ax = plt.subplots(3, 1)
    # --- Plot 1: r3_x ---
    ax[0].plot(sim_time[0:-1], g_r1, label="GLR g(k)")
    ax[0].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[0].set_title("GLR Test Statistic for r3 x")
    ax[0].legend()
    ax[0].grid(True)

    # --- Plot 2: r3_y ---
    ax[1].plot(sim_time[0:-1], g_r2, label="GLR g(k)")
    ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[1].set_title("GLR Test Statistic for r3 y")
    ax[1].legend()
    ax[1].grid(True)
    plt.xlabel("Time [s]")

    # --- Plot 2: r3_y ---
    ax[2].plot(sim_time[0:-1], g_r3, label="GLR g(k)")
    ax[2].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[2].set_title("GLR Test Statistic for r3 yaw")
    ax[2].legend()
    ax[2].grid(True)
    fig.suptitle("GLR - Observer 3")
    plt.xlabel("Time [s]")

    plt.show()






############### ------ Observer 4 - LiDAR + GYROCOMPASS ------ ###############

print("Observer 4:")
if(1):
    r_lidargyro = np.array(r_lidargyro)

    x_est4 = np.array(x_est4)
    y_est4 = np.array(y_est4)
    psi_est4 = np.array(psi_est4)
    d_true = np.array(d_true)
    d_est = np.array(d_est)
    
    mean_d = np.mean(r_lidargyro[:,0])
    std_d  = np.std(r_lidargyro[:,0])

    mean_psi = np.mean(r_lidargyro[:,1])
    std_psi  = np.std(r_lidargyro[:,1])

    #print(f"x (mean, std): {mean_x:.5f}, {std_x:.5f}")
    print(f"d (mean, std): {mean_d:.5f}, {std_d:.5f}")
    print(f"y (mean, std): {mean_psi:.5f}, {std_psi:.3f}")


    fig, axs = plt.subplots(2, 1)

    axs[0].plot(sim_time[:-1], d_true, 'k--', label='True Distance')
    axs[0].plot(sim_time[:-1], d_est, 'r-', label='Estimated Distance')
    axs[0].set_ylabel('distance [m]')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], psi_true, 'k--', label='True psi')
    axs[1].plot(sim_time[:-1], psi_est4, 'g-', label='Estimated psi')
    axs[1].set_ylabel('psi')
    axs[1].set_xlabel('Time')
    axs[1].legend()
    axs[1].grid(True)
    fig.suptitle("Estimated Values - Observer 4")
    fig.tight_layout()

    fig, axs = plt.subplots(3, 1)

    axs[0].plot(sim_time[:-1], x_true, 'k--', label='True x')
    axs[0].plot(sim_time[:-1], x_est4, 'r-', label='Estimated x')
    axs[0].set_ylabel('x')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], y_true, 'k--', label='True y')
    axs[1].plot(sim_time[:-1], y_est4, 'b-', label='Estimated y')
    axs[1].set_ylabel('y')
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(sim_time[:-1], psi_true, 'k--', label='True psi')
    axs[2].plot(sim_time[:-1], psi_est4, 'g-', label='Estimated psi')
    axs[2].set_ylabel('psi')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)

    fig.suptitle("Estimated Pose - Observer 4")
    fig.tight_layout()


    plt.figure()
    plt.plot(sim_time[0:-1],r_lidargyro[:,0],label="r_d")
    plt.plot(sim_time[0:-1],r_lidargyro[:,1],label="r_psi")
    #plt.plot(sim_time[0:-1],r_lidargyro[:,2],label="r_psi")
    plt.xlabel("Time[s]")
    plt.ylabel("residual")
    plt.legend()
    plt.title("Residuals - Observer 4")
    plt.grid(True)
    plt.tight_layout()


    glr_r4_1 = glr.GLR(WindowSize=10,residual=r_lidargyro[:,0],mean=0.0,sigma=0.0009)
    g_r1, idx_r1, mu_1_r1 = glr_r4_1.computeGlr()

    glr_r4_2 = glr.GLR(WindowSize=10,residual=r_lidargyro[:,1],mean=0.0,sigma=0.025)
    g_r2, idx_r2, mu_1_r2 = glr_r4_2.computeGlr()

    #glr_r4_3 = glr.GLR(WindowSize=10,residual=r_lidargyro[:,2],mean=0.0,sigma=0.025)
    #g_r3, idx_r3, mu_1_r3 = glr_r4_3.computeGlr()


    fig, ax = plt.subplots(2, 1)
    # --- Plot 1: r3_x ---
    ax[0].plot(sim_time[0:-1], g_r1, label="GLR g(k)")
    ax[0].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[0].set_title("GLR Test Statistic for r4 d")
    ax[0].legend()
    ax[0].grid(True)

    # # --- Plot 2: r3_y ---
    # ax[1].plot(sim_time[0:-1], g_r2, label="GLR g(k)")
    # ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    # ax[1].set_title("GLR Test Statistic for r4 y")
    # ax[1].legend()
    # ax[1].grid(True)
    # plt.xlabel("Time [s]")

    # --- Plot 2: r3_y ---
    ax[1].plot(sim_time[0:-1], g_r2, label="GLR g(k)")
    ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[1].set_title("GLR Test Statistic for r4 psi")
    ax[1].legend()
    ax[1].grid(True)
    fig.suptitle("GLR - Observer 4")
    plt.xlabel("Time [s]")

    plt.show()

print("Observer 5:")
# OBSERVER 5 LiDAR + GNSS
if(1):
    r_lidargnss = np.array(r_lidargnss)
    x_est5 = np.array(x_est5)
    y_est5 = np.array(y_est5)
    d_est5 = np.array(d_est5)
    
    mean_x = np.mean(r_lidargnss[:,0])
    std_x  = np.std(r_lidargnss[:,0])

    mean_y = np.mean(r_lidargnss[:,1])
    std_y  = np.std(r_lidargnss[:,1])

    mean_d = np.mean(r_lidargnss[:,2])
    std_d  = np.std(r_lidargnss[:,2])
    #print(f"x (mean, std): {mean_x:.5f}, {std_x:.5f}")
   
    print(f"x (mean, std): {mean_x:.5f}, {std_x:.3f}")
    print(f"y (mean, std): {mean_y:.5f}, {std_y:.3f}")
    print(f"d (mean, std): {mean_d:.5f}, {std_d:.5f}")


    fig, axs = plt.subplots(3, 1)

    axs[0].plot(sim_time[:-1], d_true, 'k--', label='True Distance')
    axs[0].plot(sim_time[:-1], d_est5, 'r-', label='Estimated Distance')
    axs[0].set_ylabel('distance [m]')
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(sim_time[:-1], x_true, 'k--', label='True x')
    axs[1].plot(sim_time[:-1], x_est5, 'b-', label='Estimated x')
    axs[1].set_ylabel('x')
    axs[1].set_xlabel('Time')
    axs[1].legend()
    axs[1].grid(True)
    fig.suptitle("Estimated Values - Observer 5")
    fig.tight_layout()

    axs[2].plot(sim_time[:-1], y_true, 'k--', label='True y')
    axs[2].plot(sim_time[:-1], y_est5, 'g-', label='Estimated y')
    axs[2].set_ylabel('x')
    axs[2].set_xlabel('Time')
    axs[2].legend()
    axs[2].grid(True)
    fig.suptitle("Estimated Values - Observer 5")
    fig.tight_layout()
    
    
    plt.figure()
    plt.plot(sim_time[0:-1],r_lidargnss[:,0],label="r_x")
    plt.plot(sim_time[0:-1],r_lidargnss[:,1],label="r_y")
    plt.plot(sim_time[0:-1],r_lidargnss[:,2],label="r_d")
    plt.xlabel("Time[s]")
    plt.ylabel("residual")
    plt.legend()
    plt.title("Residuals - Observer 5")
    plt.grid(True)
    plt.tight_layout()


    glr_r5_1 = glr.GLR(WindowSize=10,residual=r_lidargnss[:,0],mean=0.0,sigma=60)
    g_r5_1, idx_r1, mu_1_r1 = glr_r5_1.computeGlr()

    glr_r5_2 = glr.GLR(WindowSize=10,residual=r_lidargnss[:,1],mean=0.0,sigma=60)
    g_r5_2, idx_r2, mu_1_r2 = glr_r5_2.computeGlr()

    glr_r5_3 = glr.GLR(WindowSize=10,residual=r_lidargnss[:,2],mean=0.0,sigma=2)
    g_r5_3, idx_r3, mu_1_r3 = glr_r5_3.computeGlr()


    fig, ax = plt.subplots(3, 1)
    # --- Plot 1: r3_x ---
    ax[0].plot(sim_time[0:-1], g_r5_1, label="GLR g(k)")
    ax[0].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[0].set_title("GLR Test Statistic for r5 x")
    ax[0].legend()
    ax[0].grid(True)

    # --- Plot 2: r3_y ---
    ax[1].plot(sim_time[0:-1], g_r5_2, label="GLR g(k)")
    ax[1].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[1].set_title("GLR Test Statistic for r5 y")
    ax[1].legend()
    ax[1].grid(True)
    plt.xlabel("Time [s]")

    # --- Plot 2: r3_y ---
    ax[2].plot(sim_time[0:-1], g_r5_3, label="GLR g(k)")
    ax[2].axhline(8, color='red', linestyle='--', label='Threshold h')
    ax[2].set_title("GLR Test Statistic for r5 d")
    ax[2].legend()
    ax[2].grid(True)
    fig.suptitle("GLR - Observer 5")
    plt.xlabel("Time [s]")

    plt.show()
