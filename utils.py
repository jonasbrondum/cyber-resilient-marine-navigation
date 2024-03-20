import numpy as np
import open3d as o3d
from numpy.linalg import matrix_rank

extensions = [".csv", ".laz", ".las"]

def file2time(path):
    """
    **Specific to project data**
    Function takes a Path-object as input
    Returns the filename as a float Unix epoch time
    Some filenames are: xxxxx_lidar1.csv/laz/las
    Most filenames are: xxxxx.csv/laz/las
    """
    if "ENC" in str(path):
        return float(path.stem.split("_")[1])
    
    return float(path.stem.split("_")[0])

def closestIdx(array, value):
    """
    Function takes an array and a value as input
    Returns the index of the closest value in the array

    Example: Array contains timestamps, value is a timestamp
    """
    return np.argmin(np.abs(array - value))

# Function from:
# https://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-represents-a-number-float-or-int
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def filterScan(scan, params):
    """
    Function to remove outliers, and irrelevant measurement points.
    The scans are returned as (0, 0, 0) if the measurement is out of range.
    From empirical studies the cab has been found to be within the bounding_box defined below.
    
    Input:
    scan: (N x 3) numpy array
    params: Parameters-object

    Output:
    pcl: o3d.geometry.PointCloud
    """
    if len(scan) == 0:
        return None
    
    # Out of range points are removed:
    scan = scan[~np.all(scan == 0, axis=1)]

    # Points within the ship's size are filtered as they are static.
    # We filter by creating a bounding box:
    bounding_box = [-4, 8.5, -0.5, 5.5]

    ship_mask = np.logical_and.reduce((
        scan[:, 0] > bounding_box[0],
        scan[:, 0] < bounding_box[1],
        scan[:, 1] > bounding_box[2],
        scan[:, 1] < bounding_box[3] 
    ))
    
    # Apply the mask to get the filtered points
    scan = scan[~ship_mask]

    # We remove points outside the actual range of the LiDAR:
    distance_mask = np.linalg.norm(scan, axis=1) < params.lidar_max_range
    scan = scan[distance_mask]

    # If we have no points left, we return None:
    if len(scan) == 0:
        return None

    pcl = o3d.geometry.PointCloud()
    pcl.points = o3d.utility.Vector3dVector(scan)

    return pcl


# Define a function that creates a clockwise rotation matrix around the z-axis (yaw-axis)
def rotate_clockwise_yaw(yaw_degrees):
    """
    This function takes an angle in degrees and returns a clockwise rotation matrix around the z-axis (yaw-axis)

    Input:
    yaw_degrees: float (angle in degrees)

    Output:
    rotation_matrix: 3 x 3 numpy array
    """
    yaw_radians = np.radians(yaw_degrees)
    cos_yaw = np.cos(yaw_radians)
    sin_yaw = -np.sin(yaw_radians)  # Change the sign of the sine term
    rotation_matrix = np.array([[cos_yaw, -sin_yaw, 0],
                                [sin_yaw, cos_yaw, 0],
                                [0, 0, 1]])
    return rotation_matrix


def convCompassENU(compass):
    """
    This function converts between compass and ENU heading.

    Input:
    compass: float (degrees)

    Output:
    enu: float (degrees)
    """
    return (90 - compass) % 360


def skew_symmetric(v):
    """
    This function returns the skew-symmetric matrix for a vector.
    Skew symmetric form of a 3x1 vector.
    """
    return np.array(
        [[0, -v[2], v[1]],
         [v[2], 0, -v[0]],
         [-v[1], v[0], 0]], dtype=np.float64)


def checkObservability(phi_matrix, H, n_states):
    """
    This function checks for observability when a measurement is made.

    Input:
    F: n_states x n_states numpy array
    H: M x n_states numpy array

    Output:
    None
    """

    # Checking observability:     
    row_dim = H.shape[0]
    obs_matrix = np.zeros((6*row_dim, n_states))
    obs_matrix[:row_dim, :] = H
    obs_matrix[row_dim:2*row_dim, :] = H @ phi_matrix[0]
    obs_matrix[2*row_dim:3*row_dim, :] = H @ phi_matrix[1] @ phi_matrix[0]
    obs_matrix[3*row_dim:4*row_dim, :] = H @ phi_matrix[2] @ phi_matrix[1] @ phi_matrix[0]
    obs_matrix[4*row_dim:5*row_dim, :] = H @ phi_matrix[3] @ phi_matrix[2] @ phi_matrix[1] @ phi_matrix[0]
    obs_matrix[5*row_dim:6*row_dim, :] = H @ phi_matrix[4] @ phi_matrix[3] @ phi_matrix[2] @ phi_matrix[1] @ phi_matrix[0]

    print("Rank of observability matrix: ", matrix_rank(obs_matrix))
    print(np.where(np.all(obs_matrix == 0, axis=0))[0])
