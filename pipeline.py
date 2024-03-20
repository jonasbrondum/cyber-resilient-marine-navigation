import numpy as np
from pathlib import Path
import utils
from data_samples import loadData
from parameters import loadParameters
from scipy.spatial.transform import Rotation as R

from eskf import ESKF
from lidar_odometry import lidar_odometry_pipeline, correct_lidar_odometry, verifyHomogeneousTransforms
from tm_odometry import tm_odometry_pipeline


sets = [6]

def pipeline(data_path, set, fusion_mix):

    '''
    Initialize the pipeline:
    '''

    params = loadParameters()

    '''
    Loading and slicing data:
    '''
    print("Loading and slicing data...")
    data_duration = 10000
    data_offset = 0
    # Insert your data here:
    lidar_df, gps_df, imu_df, heading_df, enc_df = loadData(data_path, set, 
                                                            data_duration, data_offset)
    
    pcd_paths = np.asarray(lidar_df['path'].values)
    imu_timestamps = np.asarray(imu_df['timestamp'].values)
    imu_accel = np.asarray(imu_df[['ax', 'ay', 'az']].values)
    imu_gyro = np.asarray(imu_df[['gx', 'gy', 'gz']].values)
    
    init_heading = heading_df['heading vessel'].values[0]

    '''
    Compute LiDAR odometry:
    '''

    if "lidar" in fusion_mix:
        lidar_odom_raw, transforms_odom, _, _ = lidar_odometry_pipeline(pcd_paths, params, print_progress=True)
        verifyHomogeneousTransforms(transforms_odom)

        lidar_odom_cor = correct_lidar_odometry(lidar_odom_raw, init_heading)

        lidar_odom = np.column_stack((lidar_df['timestamp'].values, lidar_odom_cor))

        lidar_prev = np.array([0, 0, 0, init_heading])

        y_lidar = lidar_odom

    '''
    Compute Template matching odometry:
    '''
    if "tm" in fusion_mix:
        map_type = "osm" # "osm" or "enc"

        tm_odom = tm_odometry_pipeline(map_type, pcd_paths, lidar_df, gps_df, heading_df, enc_df, params)

        # Removing outliers (missing estimates = (0, 0, 0)):
        tm_mask = np.all(tm_odom[:, 1:] == 0, axis=1)
        tm_odom = tm_odom[~tm_mask]

        y_tm = tm_odom

    '''
    Perform sensor fusion (using ESKF):
    '''

    fusion = ESKF(init_heading, imu_df.shape[0])

    y_gps = np.column_stack((gps_df['timestamp'].values, gps_df[['East', 'North']].values, np.zeros([gps_df.shape[0], 1])))
    y_head = np.column_stack((heading_df['timestamp'].values, heading_df['heading vessel'].values))

    for k in range(1, len(imu_df)):
         
        # 1. Propagation of the state and covariance
        dt = imu_timestamps[k] - imu_timestamps[k - 1]
        fusion.propagate(k, dt, imu_accel[k - 1], imu_gyro[k - 1, 2])

        # 2. Measurement correction
        if "lidar" in fusion_mix:
            ###### Lidar Odometry measurements: ######
            for i in range(len(y_lidar)):
                if abs(y_lidar[i, 0] - imu_timestamps[k]) < params.imu_threshold:
                    sensor_var = params.sigma_odom

                    # Computing the measurement:
                    curr_trans = transforms_odom[i]

                    if all(v == 0 for v in curr_trans[:3, 3]):
                        continue


                    # We calculate the current rotation in the ENU frame:
                    rot_mat_enu = R.from_euler("z", utils.convCompassENU(lidar_prev[3]), degrees=True).as_matrix()

                    # We compute the delta position and heading:
                    delta_p = rot_mat_enu @ curr_trans[:3, 3]
                    delta_theta = R.from_matrix(curr_trans[:3, :3]).as_euler('zyx', degrees=True)[0]

                    # We extract the position and heading:
                    y_m = np.zeros(4)
                    y_m[:3] = lidar_prev[:3] + delta_p
                    y_m[3] = lidar_prev[3] + delta_theta

                    fusion.measurement(k, "lidar", y_m, sensor_var)

                    lidar_prev = fusion.get_measurement_matrix("lidar") @ fusion.x_hat[k]

        if "tm" in fusion_mix:
            ###### Template matching measurements: ######
            for i in range(len(y_tm)):
                if abs(y_tm[i, 0] - imu_timestamps[k]) < params.imu_threshold:
                    if map_type == "osm":
                        sensor_var = params.sigma_osm
                    elif map_type == "enc":
                        sensor_var = params.sigma_enc
                    y_m = y_tm[i, 1:]
                    fusion.measurement(k, "tm", y_m, sensor_var)
        
        if "gps" in fusion_mix:
            ###### GPS measurements: ######
            for i in range(len(y_gps)):
                if abs(y_gps[i, 0] - imu_timestamps[k]) < params.imu_threshold:
                    sensor_var = params.sigma_gps
                    y_m = y_gps[i, 1:]
                    fusion.measurement(k, "gps", y_m, sensor_var)

                    # Correcting the LiDAR odometry with GPS measurements:
                    lidar_prev = fusion.get_measurement_matrix("lidar") @ fusion.x_hat[k]
        
        if "compass" in fusion_mix:
            ###### Heading measurements: ######
            for i in range(len(y_head)):
                if abs(y_head[i, 0] - imu_timestamps[k]) < params.imu_threshold:
                    sensor_var = params.sigma_head
                    y_m = y_head[i, 1]
                    fusion.measurement(k, "heading", y_m, sensor_var)
        

        if k % 100 == 0:
            print(fusion.x_hat[k, 6])
    
    print("Pipeline finished!")


if __name__ == "__main__":

    fusion_mix = ["lidar", "gps", "compass"]

    for set in sets:
        data_path = Path.home() / "sample_data"
        pipeline(data_path, set, fusion_mix)
