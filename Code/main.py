import numpy as np
import matplotlib.pyplot as plt
import math
import pygame
import matplotlib.pyplot as plt 
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

import poseObserver
import velObserver
import imu_gnss_Observer
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

ppm = 50 #15 pixels equates to 1 meter. map will be 40m x 80m

gps_sensor = gps.GPS([x,y],[1*ppm,1*ppm]) #Create GPS sensor 1 meter std. in x and y
gyro_compas = gyroCompass.gyroCompass(psi,0.05 * math.pi/180) #Create Gyro Compas sensor 0.5 degree std.
poseObs = poseObserver.PoseObserver(x,y,psi,ppm,[0.1,0.1,0.1]) #Create Observer

sigma_u = 0.2      
sigma_v = 0.2
sigma_r = 0.2 * math.pi/180  # 0.2 deg/s
imu_sensor = imu.IMU([sigma_u, sigma_v, sigma_r])

velObs = velObserver.VelObserver(u,v,r,psi,0.2,0.2,0.2,0.6)
imuGnssObs = imu_gnss_Observer.gnssImuObserver(x,y,psi,ppm,u,v,[0.1,0.1,0.001])
#Create Vessel object
ship = vessel_kinematics.marineVessel(x,y,psi)
ship.setVelocities(u,v,r)

enviorment = env.buildEnv((800,1400)) #1200,600 pixel map.

#Sensor resolution and range
resolution = 500   #Number of points in a 360 degree scan
Sensor_range = 15 * ppm   #50 meters * pixels per meter.
print("Map Size(m): ", [3810/ppm, 2096/ppm])

laser=sensor.LaserSensor(resolution,Sensor_range,enviorment.OriginalMap,uncertainty=(0.02,0.005))
enviorment.map.fill((0,0,0))
enviorment.infomap = enviorment.map.copy()

#Reference circle location
cx, cy, radius = reference_detector.detect_circle_opencv("Map.png")
print("Center: x, y, radius in meters:", cx, cy, radius/ppm) 
print("Reference size m^2: ", math.pi*(radius/ppm)**2)
#11 Meter Radius Landmark

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

psi_meas_arr = []
psi_hat_1_arr = []

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
    #While psi is calculated as a global frame pygame y is flipped.
    psi_real = -psi

    x_measured, y_measured, course_over_ground = gps_sensor.gpsPose(dt,[x,y])
    psi_meassured = wrap_pi(gyro_compas.psi_measured(psi,fault_compas_bias,fault_gyrocompas))
    psi_meas_arr.append(psi_meassured)

    u_meas, v_meas, r_meas = imu_sensor.measure([u,v,r],fault_imu_var,fault_imu_bias)

    if n <= 0:
        x_hat1 = x_measured
        y_hat1 = y_measured
        psi_hat1 = psi_meassured
    else:
        x_hat1, y_hat1, psi_hat1 = poseObs.step(dt, u,v,r, x_measured,y_measured,psi_meassured)
    epsilon_1 = [(x_measured - x_hat1)/ppm, (y_measured - y_hat1)/ppm, wrap_pi(psi_meassured - psi_hat1)]
    r_poseObs.append(epsilon_1)
    psi_hat_real = -psi_hat1
    psi_hat_1_arr.append(psi_hat1)
    #Velocity based observer
    u_hat, v_hat, r_hat, psi_hat2 = velObs.step(dt, u, v, r, u_meas, v_meas, r_meas,  psi_meassured)
    epsilon_2 = [(u_meas - u_hat),(v_meas-v_hat),(r_meas-r_hat), wrap_pi(psi_meassured - psi_hat2)]
    r_velObs.append(epsilon_2)


    #IMU GNSS OBSERVER
    x_hat3,y_hat3,psi3 = imuGnssObs.step(dt,u_meas,v_meas,r_meas,x_measured,y_measured,course_over_ground)
    epsilon_3 = [(x_measured-x_hat3)/ppm,(y_measured-y_hat3)/ppm]
    r_imuGnss.append(epsilon_3)

    # --- Update LiDAR position with true pose --- #
    laser.position = (int(x), int(y))
    #print(laser.position)
    fault_distance = True if sim_time[n] > 300 else False
    fault_angle    = True if sim_time[n] > 300 else False
   

    sensor_data = laser.senseObstacle(fault_distance,fault_angle) #Raw sensor data
    #Save to global frame for plotting in map.
    if sensor_data:
        enviorment.dataStorage(sensor_data)
    enviorment.show_sensorData()
    enviorment.map.blit(enviorment.infomap, (0,0))

    #Take point cloud and associate the meassrued points to the known landmark. Done by taking matching point positions to landmark position
    pointCloudGlobal = point_matching.pointCloud_screen_global(sensor_data)
    associations = point_matching.match_points([(cx,cy)],pointCloudGlobal,1,radius,ppm)
    
    #Meassured
    if not associations:
        print("No Points Associated to Land Mark")
    else:
        c_hat = point_matching.meassured_circle(associations,radius)
        meassured_dist  = point_matching.meassured_distance((x,y),c_hat,ppm)
        meassured_theta = point_matching.meassured_bearing((x,y,psi_real),c_hat)
        theta_meas_arr = np.append(theta_meas_arr,meassured_theta)
        #Model
        dist_to_landmark = point_matching.expected_distance((x_hat1,y_hat1),(cx,cy),radius,ppm)
        theta = point_matching.expected_bearing((x_hat1,y_hat1,psi_hat_real),(cx,cy)) #We pass the inverted heading to match the visual expectation of the sim
        theta_arr = np.append(theta_arr,theta)

        r1 = np.append(r1, (meassured_dist - dist_to_landmark))
        #Theta represents an angel make sure no wrap around spikes occur when comparing them, theta meassure and model theta are 2pi and 0
        r2 = np.append(r2, point_matching.wrap_pi(meassured_theta - theta))         #print("r1 (in m): ", r1 )
        #print("r2: ", r2)
        #print("Distance: ", dist_to_landmark)
    bow_x = x + L * np.cos(psi)
    bow_y = y + L * np.sin(psi)

    bow_x_hat = x_hat1 + L * np.cos(psi_hat1)
    bow_y_hat = y_hat1 + L * np.sin(psi_hat1)

    bow_x_hat3 = x_hat3 + L * np.cos(psi3)
    bow_y_hat3 = y_hat3 + L * np.sin(psi3)

    
    #Draw ship
    pygame.draw.line(enviorment.map, (0,255,0), (x,y), (bow_x,bow_y), 5)
    #GNSS Gyrocompass
    pygame.draw.line(enviorment.map, (0,255,255), (x_hat1,y_hat1), (bow_x_hat,bow_y_hat), 5)
    #IMU GNSS
    pygame.draw.line(enviorment.map, (255,255,0), (x_hat3,y_hat3), (bow_x_hat3,bow_y_hat3), 5)


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
        u = -1.5                   # forward m/s
        v = -0.0                   # Sideway m/s
        r = -10 * math.pi/180      # Yaw rate angular velocity rad/s
        ship.setVelocities(u,v,r)
    elif(sim_time[n] < 65):
        u = -1.0
        v=0.0
        r=0.0
        ship.setVelocities(u,v,r)
    elif(sim_time[n] < 75):
         #fault_imu_var = [0, 1, 0]
         #fault_imu_bias = [0, 0.5, 0]
         u=0.0
         v=1.2
         r=0.0
         ship.setVelocities(u,v,r)
    elif(sim_time[n] < 90):
         u = 1.0
         v = 0.0
         r = 0.0
         ship.setVelocities(u,v,r)        
    elif(sim_time[n] < 110):
        fault_gyrocompas = 3.5 * math.pi/180
        #fault_compas_bias = 3 * math.pi/180
        #gps_sensor.setUncertainty([1*ppm,1*ppm],150)
        #fault_imu_bias = [0.5, 0.5, 0]
        #fault_imu_var = [1, 0, 0]
        #fault_imu_bias = [0, 0, 0]
        u = 0.0
        v = -0.4
        r = 0.0
        ship.setVelocities(u,v,r)
    elif(sim_time[n] < 160):
        u = 0.0
        v= 0.0
        r = 10 * math.pi/180
        ship.setVelocities(u,v,r )
    elif(sim_time[n] < 170):
        #fault_imu_var = [0, 0, 1]
        #fault_imu_bias = [0.5, 0.5, 0.5]
        u = 0.0
        v = 0.0
        r = 0
        ship.setVelocities(u,v,r )

    if(sim_time[n] > 200):
        running = False
    #print("Simulatedtime:", sim_time)
    n+=1

print("Samples: ", n)

r_poseObs = np.array(r_poseObs)
r_velObs = np.array(r_velObs)
r_imuGnss = np.array(r_imuGnss)

############### ------ Observer 1 ------ ###############

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

plt.figure()
plt.plot(sim_time[0:-1],r_poseObs[:,0],label="r_x")
plt.plot(sim_time[0:-1],r_poseObs[:,1],label="r_y")
plt.plot(sim_time[0:-1],r_poseObs[:,2],label="r_psi")
plt.xlabel("Time[s]")
plt.ylabel("residual")
plt.legend()
plt.grid(True)
plt.tight_layout()

glr_r1_1 = glr.GLR(WindowSize=30,residual=r_poseObs[:,0],mean=0.0,sigma=0.93)
g_r1, idx_r1, mu_1_r1 = glr_r1_1.computeGlr()

glr_r1_2 = glr.GLR(WindowSize=30,residual=r_poseObs[:,1],mean=0.0,sigma=0.94)
g_r2, idx_r2, mu_1_r2 = glr_r1_2.computeGlr()

glr_r1_3 = glr.GLR(WindowSize=30,residual=r_poseObs[:,2],mean=0.0,sigma=0.00012)
g_r3, idx_r3, mu_1_r3 = glr_r1_3.computeGlr()


fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
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
# Improve layout
plt.xlabel("Time [s]")



############### ------ Observer 2 ------ ###############
print("Observer 2: ")
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
plt.tight_layout()


glr_r2_1 = glr.GLR(WindowSize=30,residual=r_velObs[:,0],mean=0.0,sigma=0.2)
g_r1, idx_r1, mu_1_r1 = glr_r2_1.computeGlr()

glr_r2_2 = glr.GLR(WindowSize=30,residual=r_velObs[:,1],mean=0.0,sigma=0.2)
g_r2, idx_r2, mu_1_r2 = glr_r2_2.computeGlr()

glr_r2_3 = glr.GLR(WindowSize=30,residual=r_velObs[:,2],mean=0.0,sigma=0.003)
g_r3, idx_r3, mu_1_r3 = glr_r2_3.computeGlr()

glr_r2_4 = glr.GLR(WindowSize=30,residual=r_velObs[:,3],mean=0.0,sigma=0.001)
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




############### ------ Observer 3 ------ ###############

print("Observer 3:")
mean_x = np.mean(r_imuGnss[:,0])
std_x  = np.std(r_imuGnss[:,0])


mean_y = np.mean(r_imuGnss[:,1])
std_y  = np.std(r_imuGnss[:,1])

print(f"x (mean, std): {mean_x:.5f}, {std_x:.5f}")
print(f"y (mean, std): {mean_y:.5f}, {std_y:.5f}")



plt.figure()
plt.plot(sim_time[0:-1],r_imuGnss[:,0],label="r_x")
plt.plot(sim_time[0:-1],r_imuGnss[:,1],label="r_y")
plt.xlabel("Time[s]")
plt.ylabel("residual")
plt.legend()
plt.grid(True)
plt.tight_layout()


glr_r3_1 = glr.GLR(WindowSize=30,residual=r_imuGnss[:,0],mean=0.0,sigma=0.99)
g_r1, idx_r1, mu_1_r1 = glr_r3_1.computeGlr()

glr_r3_2 = glr.GLR(WindowSize=30,residual=r_imuGnss[:,1],mean=0.0,sigma=0.99)
g_r2, idx_r2, mu_1_r2 = glr_r3_2.computeGlr()

fig, ax = plt.subplots(2, 1, figsize=(10, 10), sharex=True)
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


psi_meas_arr = np.array(psi_meas_arr)
psi_hat_1_arr = np.array(psi_hat_1_arr)
plt.figure()
plt.plot(sim_time[0:-1],(psi_meas_arr-psi_hat_1_arr))
plt.title("Psi Hat 1")
plt.show()


# fig, ax = plt.subplots(2, 2)

# ax[0,0].plot(sim_time[0:-1],r1), ax[0,0].set_title("Residual r1")
# ax[0,0].grid(True)

# r1_mean = np.mean(r1)
# r1_var  = np.var(r1)
# ax[1,0].hist(r1)
# ax[1,0].set_title(f"Histogram of r1 (mean={r1_mean:.3f}, σ^2={r1_var:.3f})")
# ax[1,0].axvline(r1_mean, linestyle='--', label=f"mean")
# ax[1,0].axvline(r1_mean + np.sqrt(r1_var), color='red', linestyle=':', label=r"$\mu \pm \sigma$")
# ax[1,0].axvline(r1_mean - np.sqrt(r1_var), color='red', linestyle=':')
# ax[1,0].legend()


# ax[0,1].plot(sim_time[0:-1],r2) , ax[0,1].set_title("Residual r2")
# ax[0,1].grid(True)

# r2_mean = np.mean(r2)
# r2_var  = np.var(r2)
# print("r2 mean: ", r2_mean)
# print("r2 var: ", r2_var)

# ax[1,1].hist(r2)
# ax[1,1].set_title(f"Histogram of r2 (mean={r2_mean:.5f}, σ^2={r2_var:.5f})")
# ax[1,1].axvline(r2_mean, linestyle='--', label=f"mean")
# ax[1,1].axvline(r2_mean + np.sqrt(r2_var), color='red', linestyle=':', label=r"$\mu \pm \sigma$")
# ax[1,1].axvline(r2_mean - np.sqrt(r2_var), color='red', linestyle=':')

# fig.suptitle("Residual", fontsize=16)  # main title
# plt.tight_layout()

# # plt.figure()
# # plt.plot(sim_time[0:-1],theta_meas_arr,label="Theta Meas")
# # plt.plot(sim_time[0:-1],theta_arr,label="Theta")
# # plt.legend()
# # plt.show()



# s = ctrl.tf('s')

# zeta = 0.7
# w0 = 1.0

# H = (w0**2) / (s**2 + 2*zeta*w0*s + w0**2)

# t_out, r1_filt = ctrl.forced_response(H, T=sim_time[0:-1], U=r1)
# t_out, r2_filt = ctrl.forced_response(H, T=sim_time[0:-1], U=r2)
# r2_filt = np.asarray(r2_filt, dtype=float)

# plt.figure()
# plt.plot(t_out,r1_filt,label="R1 Filtered")
# plt.plot(sim_time[0:-1],r1,label="R1")
# plt.plot(t_out,r2_filt,label="R2 Filtered")
# plt.plot(sim_time[0:-1],r2,label="R2")
# plt.legend()

# #Mean r1: 0.1
# #Sigma^2 r1: 0.002

# #Mean r2: -0.0018
# #Sigma^2 r2: 0.00002

# m1 = 10 
# glr_r1 = glr.GLR(WindowSize=m1,residual=r1_filt,mean=0.1,sigma=math.sqrt(0.002))
# g_r1, idx_r1, mu_1_r1 = glr_r1.computeGlr()


# m2 = 10
# glr_r2 = glr.GLR(WindowSize=25,residual=r2_filt,mean=0,sigma=math.sqrt(0.00002))
# g_r2, idx_r2, mu_1_r2 = glr_r2.computeGlr()


# fig, ax = plt.subplots(2, 1)

# # Top subplot: r2 + mu1
# ax[0].plot(t_out, mu_1_r2, label="μ₁")
# ax[0].plot(t_out, r2_filt, label="r₂ filtered")
# ax[0].set_title("Residual r₂ and residual mean μ₁")
# ax[0].legend()
# ax[0].grid(True)

# # Bottom subplot: GLR statistic
# ax[1].plot(t_out, g_r2, label="GLR g(k)")
# ax[1].axhline(50, color='red', linestyle='--', label='Threshold h')
# ax[1].set_title("GLR Test Statistic for r₂")
# ax[1].legend()
# ax[1].grid(True)


# fig, ax = plt.subplots(2, 1)

# # Top subplot: r1 + mu1
# ax[0].plot(t_out, mu_1_r1, label="μ₁")
# ax[0].plot(t_out, r1_filt, label="r₁ filtered")
# ax[0].set_title("Residual r₁ and residual mean μ₁")
# ax[0].legend()
# ax[0].grid(True)

# # Bottom subplot: GLR statistic
# ax[1].plot(t_out, g_r1, label="GLR g(k)")
# ax[1].axhline(50, color='red', linestyle='--', label='Threshold h')
# ax[1].set_title("GLR Test Statistic for r₁")
# ax[1].legend()
# ax[1].grid(True)

# plt.tight_layout()
# plt.show()