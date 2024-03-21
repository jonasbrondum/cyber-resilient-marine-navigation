# Cyber-resilient multi-modal sensor fusion for autonomous shipnavigation

This repository is a selection of code used in my master thesis with the title: **Cyber-resilient multi-modal sensor fusion for autonomous shipnavigation**. I've tried to clean the code and make it as readable as possible. Hopefully, this can be useful for anyone wanting to implement their own error-state kalman filter.

## Project abstract
Ensuring the integrity of navigation sensors is paramount for safe maritime navigation. Un- fortunately, cyber-attacks targeting satellite positioning information and onboard sensors have exposed vulnerabilities, posing a significant threat to navigation safety. In near-coast and urban environments with high traffic density and static structures, the risk of collisions and grounding is elevated. Without a navigator’s sanity check of crucial information such as position, heading, and velocity, cyber attacks could lead to accidents by compromising accurate data.

This thesis addresses the challenge of enhancing the cyber-resilience of maritime surface vessel navigation systems. Various sensor systems are evaluated, and auxiliary sensing methods are proposed to create a GNSS-independent navigation system. The thesis introduces two methods—point cloud registration and template matching—for an auxiliary positioning system. Additionally, an observer is designed to leverage the existing array of sensors within a framework supported by auxiliary measurements.

The developed navigation system is tested using real-world data collected in Copen- hagen’s inner harbor. Despite the absence of GNSS measurements, the system pro- duces sufficiently accurate positioning. The resulting position estimate demonstrates the cyber-resilience of the proposed solution for GNSS-independent navigation.

## Contents

The project consisted of several methods for calculating an odometry to use in localization.
1. LiDAR odometry from point cloud registration. This is located in `lidar_odometry.py`.
2. Template matching (TM) odometry using LiDAR with electronic navigational charts (ENC) or open-street map (OSM). This is located in `tm_odometry.py`.

These odometries were along with GPS, Compass, and IMU measurements used in the error-state Kalman filter for continuous localization. The error-state kalman filter is implemented as a class in `eskf.py`.

![Project pipeline](/imgs/full_pipeline.png "Project pipeline")

The pipeline is combined in `pipeline.py`

## Dataset

The pipeline as it is implemented here makes use of pickled DataFrames. The data used for the project is not for public use, so it is, unfortunately, unavailable for me to share. The `loadData()`in `data_samples.py` can easily be modified to fit other data storage.

The dataset was partitioned into smaller sets for development such that the relative East-North-Up coordinate system could be utilized.

The current structure accomodates 5 DataFrames: 
### `lidar_df`

| `timestamp`       | `path`               |
|-------------------|----------------------|
| Unix Epoch format | path to `.pcd`-files |

### `gps_df`

| `timestamp`          | `latitude` | `longitude ` | `East` | `North` | `sog` | `cog` |
|----------------------|---|---|---|---|---|---|
| in Unix Epoch format | - | - | - | - | speed over ground | course over ground |

### `imu_df`

| `timestamp` | `q0` | `q1` | `q2` | `q3` | `ax` | `ay` | `az` | `gx` | `gy` | `gz` | 
|---|---|---|---|---|---|---|---|---|---|---|
| in Unix Epoch format | quaternions | acceleration | angular velocities |

### `heading_df`

| `timestamp` | `heading uncorrected` | `heading vessel` | 
|---|---|---|
| in Unix Epoch format | - | - |

### `enc_df`

| `timestamp` | `path` | `enc_max_range` | `latitude` | `longitude` | `pxwidth` |  
|---|---|---|---|---|---|
| in Unix Epoch format | path to enc-generated images | max range in meters | center of image | center of image | pixelwidth of the image |

## How-to

Project requirements can be installed from `requirements.txt`. 

`pipeline.py` is set up to run with the project parameters.

*Note*: The parameters used for the different sections of the pipeline are mostly defined in the `parameters.py`class with the call of `loadParameters()`.

## Quality statement 

This is research quality code, not suitable for production or any real use case. The code is provided as-is with no guarantees or warranties.

