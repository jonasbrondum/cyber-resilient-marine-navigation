import numpy as np

"""
The Parameters class is used to store the parameters for the different sections of the pipeline:
1. Point cloud filtering
2. Pairwise registration
3. Image generation from the map
4. Sensor fusion using the ESKF
5. Plotting
"""


def loadParameters():

    params = Parameters()

    """
    For filtering:
    """
    # These thresholds are in meters for filtering the point clouds

    lidar_max_range = 170

    """
    For the pairwise registration:
    """
    voxel_size = 1
    voxel_size_down = 1.5 # From optimization: 1.4
    # This threshold can be made dynamic!
    max_nn_normals = 30 # For the normal estimation of the global registration
    max_nn_fpfh = 100 # For the FPFH estimation of the global registration
    radius_normal_global = voxel_size_down * 2 # For the normal estimation of the global registration
    radius_feature_global = voxel_size_down * 5 # For the FPFH estimation of the global registration
    distance_factor_coarse = 1.5 # For the global registration # From optimization: 1.5
    distance_factor_fine = 0.7 # For the local registration # From optimization: 0.5
    radius_normal_local = 10 # For the normal estimation of the local registration
    max_nn_normal_local = 10 # For the normal estimation of the local registration
    distance_threshold_global = voxel_size_down * distance_factor_coarse # For the global registration
    distance_threshold_local = voxel_size_down * distance_factor_fine # For the local registration
    loss_sigma = 4.52 # 4.52 Geman-McClure loss sigma from paper 
    
    """
    For the map:
    """
    occlusion_angle_threshold = 15
    cluster_min_size = 65

    """
    For the sensor fusion:
    """
    # We define static variables:
    grav_acceleration = 9.815476190 # [m/s^2] From paper about gravity measurements in Copenhagen
    g = np.array([0, 0, grav_acceleration]) # Gravity vector
    imu_threshold = 0.0015 # Time delta for new correction measurements: 400 Hz -> dt = 2.5 ms so we go a bit below at: 1.5 ms
    
    # From datasheet we get initial bias error:
    acc_b_0 = 0.005*grav_acceleration
    gyr_b_0 = 0.2/180*np.pi # rad/s


    # Measurement noise sigmas from datasheet:
    # Compass
    sigma_head = np.diag([0.4]) # From datasheet

    # GPS
    # For z measurement in gps we could use another sigma: 1.7*5/sqrt(2)/2 = 3.005
    # Source: https://junipersys.com/support/article/6614
    sigma_gps = np.diag([1.768, 1.768, 1.7*1.768]) # From datasheet we have the 2dRMS which is 2*sqrt(2)*sigma_gps so -> sigma_gps = 5/sqrt(2)/2
    sigma_gps_v = np.diag([1.768, 1.768, 1.7*1.768, 0.05144, 0.05144, 1.7*0.05144])

    # IMU
    sigma_imu_f = 0.000060*grav_acceleration # From datasheet we get noise density: 0.000060 # m/s^2/sqrt(Hz)
    sigma_imu_w = 0.03/180*np.pi # From datasheet we get noise density: 0.03 # deg/s/sqrt(Hz)
    sigma_imu_f_bias = 0.005*grav_acceleration # From datasheet we get an initial bias error, however this might not be the right value
    sigma_imu_w_bias = 0.2/180*np.pi # From datasheet we get an initial bias error, however this might not be the right value


    # From numerical approximation using numerical_variance.py
    # OSM
    sigma_osm = np.diag([7.2740, 8.6320, 0]) # From numerical approximation
    # ENC
    sigma_enc = np.diag([11.7029, 12.9056, 0]) # From numerical approximation
    # LiDAR odometry
    sigma_odom = np.diag([7.1704, 13.5951, 1.9778, 10.6255]) # From numerical approximation [m, m, m, deg]
    

    # Process noise sigmas (experimentally determined):
    sigma_vn = 3.0 # Velocity
    sigma_thetan = 0.113 # Heading
    sigma_aw = 1.5 # acceleration
    sigma_ww = 1.0 # angular velocity  
    sigma_ln = 1.0 # lidar odometry bias

    # Lidar odometry bias:
    tau_l = 10 # From book is 60.
    lambda_l = 1/tau_l # From book is 1/60.
    sigma_lb = 5.0 # From book is 3.

    # Conversion factor from nautical miles per hour to meters per second
    naumph2mps = 0.514444444 # 1 knot = 0.514444444 m/s

    n_states = 14

    # Process noise jacobian:
    F_i = np.zeros([n_states, 11])
    F_i[3:6, 0:3] = np.eye(3) # Velocity impulse
    F_i[6:7, 3:4] = np.eye(1) # Angular/heading impulse
    F_i[7:10, 4:7] = np.eye(3) # Acceleration bias impulse
    F_i[10:11, 7:8] = np.eye(1) # Gyro bias impulse
    F_i[11:14, 8:11] = np.eye(3) # Lidar Odometry bias impulse

    # Initial covariance:
    P_init = np.eye(n_states)
    # Position and velocity:
    P_init[0:6, 0:6] = sigma_gps_v**2
    # Heading:
    P_init[6, 6] = sigma_head**2
    # acc bias:
    P_init[7:10, 7:10] = np.eye(3) * sigma_imu_f**2
    # gyr bias:
    P_init[10:11, 10:11] = np.eye(1) * sigma_imu_w**2
    # lidar bias:
    P_init[11:14, 11:14] = np.eye(3) * sigma_lb**2


    # Measurement matrices:
    
    # Position and velocity:
    H_pv = np.zeros([6, n_states])
    # position
    H_pv[:3, :3] = np.eye(3)
    # velocity
    H_pv[3:6, 3:6] = np.eye(3)

    # Position measurement:
    H_gps = np.zeros((3, n_states))
    H_gps[:, :3] = np.eye(3)

    # Heading measurement:
    H_head = np.zeros([1, n_states])
    H_head[0, 6] = 1

    # Lidar odometry measurement:
    H_lidar = np.zeros([4, n_states])
    H_lidar[:3, :3] = np.eye(3)
    H_lidar[:3, -3:] = np.eye(3)
    H_lidar[3, 6] = 1

    # TM measurement:
    H_tm = np.zeros([3, n_states])
    H_tm[:, :3] = np.eye(3)

    """
    For plotting:
    """
    point_size = 1

    params.add_parameters(lidar_max_range=lidar_max_range,
                            voxel_size=voxel_size,
                            voxel_size_down=voxel_size_down,
                            max_nn_normals=max_nn_normals,
                            max_nn_fpfh=max_nn_fpfh,
                            radius_normal_global = radius_normal_global,
                            radius_feature_global = radius_feature_global,
                            distance_factor_coarse = distance_factor_coarse,
                            distance_factor_fine = distance_factor_fine,
                            radius_normal_local = radius_normal_local,
                            max_nn_normal_local = max_nn_normal_local,
                            distance_threshold_global = distance_threshold_global,
                            distance_threshold_local = distance_threshold_local,
                            loss_sigma = loss_sigma,
                            occlusion_angle_threshold=occlusion_angle_threshold,
                            cluster_min_size=cluster_min_size,
                            acc_b_0=acc_b_0,
                            gyr_b_0=gyr_b_0,
                            g=g,
                            imu_threshold=imu_threshold,
                            sigma_head=sigma_head,
                            sigma_osm=sigma_osm,
                            sigma_enc=sigma_enc,
                            sigma_odom=sigma_odom,
                            sigma_gps=sigma_gps,
                            sigma_gps_v=sigma_gps_v,
                            sigma_imu_f=sigma_imu_f,
                            sigma_imu_w=sigma_imu_w,
                            sigma_imu_f_bias=sigma_imu_f_bias,
                            sigma_imu_w_bias=sigma_imu_w_bias,
                            sigma_vn=sigma_vn,
                            sigma_thetan=sigma_thetan,
                            sigma_aw=sigma_aw,
                            sigma_ww=sigma_ww,
                            sigma_ln=sigma_ln,
                            lambda_l=lambda_l,
                            sigma_lb=sigma_lb,
                            naumph2mps=naumph2mps,
                            n_states=n_states,
                            F_i=F_i,
                            P_init=P_init,
                            H_pv=H_pv,
                            H_gps=H_gps,
                            H_head=H_head,
                            H_lidar=H_lidar,
                            H_tm=H_tm,
                            point_size=point_size)

    return params


class Parameters:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __str__(self):
        return str(self.__dict__)
    
    def add_parameters(self, **kwargs):
        self.__dict__.update(kwargs)