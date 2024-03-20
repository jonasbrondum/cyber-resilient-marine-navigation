import pandas as pd
from pathlib import Path
import utils
import open3d as o3d
from pymap3d.enu import geodetic2enu
import numpy as np
from skimage.measure import block_reduce

# We will define the indices for the sets we will use
idx_set1 = (50, 250)
idx_set2 = (250, 450)
idx_set3 = (2500, 2700)
idx_set4 = (10, 2700)
idx_set5 = (0, 500)
idx_set6 = (500, 1000)
idx_set7 = (1000, 3000)
idx_set8 = (3000, 6000)

# Set set to 0 if you don't want to process that set
idx_sets = [idx_set1, idx_set2, idx_set3, idx_set4, idx_set5, idx_set6, idx_set7, idx_set8]





def loadData(path, set, duration, offset):
    """
    This function assumes that data is stored safely in the specified path.
    It also assumes it's a dataframe.

    The function loads the data and then slices it according to the specified duration and offset
    """ 

    # Loading data:
    
    lidar_df = pd.read_pickle(str(path / f'set_{set}' / f'lidar_df_set{set}.pkl'))
    gps_df = pd.read_pickle(str(path / f'set_{set}' / f'gps_df_set{set}.pkl'))
    imu_df = pd.read_pickle(str(path / f'set_{set}' / f'imu_df_set{set}.pkl'))
    heading_df = pd.read_pickle(str(path / f'set_{set}' / f'heading_df_set{set}.pkl'))
    enc_df = pd.read_pickle(str(path / f'set_{set}' / f'enc_df_set{set}.pkl'))

    # Slicing data:
    time_set_start = lidar_df['timestamp'].values[0]
    time_set_end = lidar_df['timestamp'].values[-1]

    # We ensure duration and offset don't exceed the bounds of the data set:
    duration_slice = duration if time_set_end - time_set_start > duration else time_set_end - time_set_start
    time_start = time_set_start + offset if time_set_start + offset <= time_set_end else time_set_start
    time_end = time_start + duration_slice if time_start + duration_slice <= time_set_end else time_set_end
    
    print(f"Slice of data set is: {time_start} [s] - {time_end} [s]")
    
    
    lidar_df = lidar_df[(lidar_df['timestamp'] >= time_start) & (lidar_df['timestamp'] <= time_end)]
    gps_df = gps_df[(gps_df['timestamp'] >= time_start) & (gps_df['timestamp'] <= time_end)]
    gps_df[['East', 'North']] = gps_df[['East', 'North']].values - gps_df[['East', 'North']].values[0,:]
    imu_df = imu_df[(imu_df['timestamp'] >= time_start) & (imu_df['timestamp'] <= time_end)]
    heading_df = heading_df[(heading_df['timestamp'] >= time_start) & (heading_df['timestamp'] <= time_end)]
    enc_df = enc_df[(enc_df['timestamp'] >= time_start) & (enc_df['timestamp'] <= time_end)]

    return lidar_df, gps_df, imu_df, heading_df, enc_df
        
    
