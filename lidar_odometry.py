import numpy as np
import open3d as o3d
import utils
from scipy.spatial.transform import Rotation as R


def verifyPointCloudPairwiseSize(source, target, min_size=20):
    """
    This function verifies that the point clouds are large enough to be registered.

    Input:
    source: o3d.geometry.PointCloud
    target: o3d.geometry.PointCloud
    min_size: int

    Output:
    bool
    """
    return len(np.asarray(source.points)) > min_size and len(np.asarray(target.points)) > min_size

def updateCurrentPose(global_transform):
    """
    This function takes a "global" homogeneous transform matrix and an initial heading for that "global" transform.
    It returns the current pose in ENU coordinates.
    BE AWARE: The output heading is relative!

    Input:
    global_transform: 4x4 numpy array (Homogeneous transformation matrix)
    initial_heading_enu: float (degrees)

    Output:
    current_pose: 1x6 numpy array (East, North, Up, roll, pitch, yaw)
    """

    xyz = global_transform[:3,3].reshape(3)
    ypr = R.from_matrix(global_transform[:3,:3]).as_euler('zyx', degrees=True).reshape(3)
    curr_pose = np.concatenate((xyz, ypr[::-1] % 360), dtype=np.float64) 

    return curr_pose


def preprocessPointCloud(pcl, params):
    """
    A function to preprocess a point cloud.
    Takes as input: o3d.geometry.PointCloud
    Returns: o3d.geometry.PointCloud
    """
    # We unpack the parameters:
    voxel_size_down = params.voxel_size_down
    max_nn_normals = params.max_nn_normals
    max_nn_fpfh = params.max_nn_fpfh
    radius_normal = params.radius_normal_global
    radius_feature = params.radius_feature_global

    # We downsample the point cloud:
    pcl_down = pcl.voxel_down_sample(voxel_size_down)

    # We estimate the normals:
    pcl_down.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=max_nn_normals))

    # We compute the FPFH features:
    pcl_fpfh = o3d.pipelines.registration.compute_fpfh_feature(pcl_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=max_nn_fpfh))

    return pcl_down, pcl_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, params):
    """
    This function performs a global (RANSAC) registration between two point clouds.

    Input:
    source_down: o3d.geometry.PointCloud
    target_down: o3d.geometry.PointCloud
    source_fpfh: o3d.geometry.PointCloud
    target_fpfh: o3d.geometry.PointCloud
    voxel_size: float

    Output:
    result: o3d.pipelines.registration.RegistrationResult
    """
    # We unpack the parameter:
    distance_threshold_global = params.distance_threshold_global
    
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold_global,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold_global)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result



def refine_registration(source, target, source_fpfh, target_fpfh, result_ransac, params):
    # We unpack the parameters:
    distance_threshold_local = params.distance_threshold_local
    sigma = params.loss_sigma
    loss = o3d.pipelines.registration.GMLoss(k=sigma)

    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold_local, result_ransac.transformation,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(loss))
    return result



def pairwiseRegistration(source_path, target_path, params):
    """
    Function to perform pairwise registration between two point clouds.
    The function performs both global (RANSAC) and local (GeneralizedICP) registration.
    
    Input:
    source_path: Path-object
    target_path: Path-object
    params: Parameters-object

    Output:
    result: o3d.pipelines.registration.RegistrationResult
    target: o3d.geometry.PointCloud
    """
    # Parameters:
    max_nn_normal_local = params.max_nn_normal_local
    radius_normal_local = params.radius_normal_local   

    # Load the point clouds:
    source = o3d.io.read_point_cloud(str(source_path))
    target = o3d.io.read_point_cloud(str(target_path))

    # Preprocess the point clouds:
    source_filtered = utils.filterScan(np.asarray(source.points), params)
    target_filtered = utils.filterScan(np.asarray(target.points), params)

    # We check if point cloud is too small:
    if source_filtered == None or target_filtered == None or not verifyPointCloudPairwiseSize(source_filtered, target_filtered):
        return None, None

    # We perform the global registration:
    source_down, source_fpfh = preprocessPointCloud(source_filtered, params)
    target_down, target_fpfh = preprocessPointCloud(target_filtered, params)
    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, params)

    source_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=max_nn_normal_local))
    target_filtered.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal_local, max_nn=max_nn_normal_local))
    
    source_filtered.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN())
    target_filtered.estimate_covariances(search_param=o3d.geometry.KDTreeSearchParamKNN())

    # We perform the local registration:
    result = refine_registration(source_filtered, target_filtered, source_fpfh, target_fpfh, result_ransac, params)
    
    return result, target_filtered



def lidar_odometry_pipeline(pcd_paths, params, print_progress=True):

    """
    This function takes a list of point cloud paths and returns the poses, transforms, inlier_rmses and fitnesses.

    Input:
    pcd_paths: list of paths
    params: Parameters-object
    print_progress: bool

    Output:
    poses: numpy array (N x 6)
    transforms: numpy array (N x 4 x 4)
    inlier_rmses: numpy array (N x 1)
    fitnesses: numpy array (N x 1)
    """

    init_trans = np.identity(4)

    sequence_transform = np.identity(4)  # Placeholder value

    # Poses are: [x, y, z, roll, pitch, yaw]
    poses = np.zeros((pcd_paths.shape[0], 6))
    transforms = np.zeros((pcd_paths.shape[0], 4, 4))
    inlier_rmses = np.zeros(pcd_paths.shape[0])
    fitnesses = np.zeros(pcd_paths.shape[0])

    transforms[0] = init_trans

    # We iterate over the point clouds as source and target:
    for i, (source_path, target_path) in enumerate(zip(pcd_paths[:-1], pcd_paths[1:])):
        if print_progress: print(f"Processing pair {i+1}/{len(pcd_paths)-1}")
        """
        Pairwise Registration (using global and local ICP):
        """

        result, target_filtered = pairwiseRegistration(source_path, target_path, params)
        
        if result is None and target_filtered is None:
            print("One of the point clouds is too sparse. Skipping pair. \n")
            continue

        curr_transform = result.transformation
        if result.fitness == 0 and result.inlier_rmse == 0:
            print("NaN transform")
            curr_transform = np.identity(4)
        
        # We print evaluation:
        if print_progress: print(f"Fitness: {result.fitness:.03f} -- Inlier RMSE: {result.inlier_rmse:.03f}")

        # We update the local map transformation
        sequence_transform = sequence_transform @ curr_transform

        # We update the current pose:
        poses[i+1] = updateCurrentPose(sequence_transform)

        if i % 10 == 0 and print_progress:
            print(f"Current pose: {poses[i+1]}")

        
        # Finally we update the pose, transform and inlier_rmse:
        transforms[i+1] = np.asarray(curr_transform)
        inlier_rmses[i+1] = result.inlier_rmse
        fitnesses[i+1] = result.fitness

        # We initialize the next registration with the current transformation
        init_trans = curr_transform

    print("Registration done!")
    return poses, transforms, inlier_rmses, fitnesses

def verifyHomogeneousTransforms(transforms):
    """
    Function to verify that the transformations are homogeneous.

    Input:
    transforms: list of 4x4 numpy arrays ("Homogeneous" transformation matrices)
    """
    
    # We verify that the transformations returned are indeed homogeneous:
    homo_bool = True
    for i, transform in enumerate(transforms):
        inv = transform @ np.linalg.inv(transform)
        if not np.allclose(inv, np.identity(4)):
            print(f'{i} - Not homogeneous')
            homo_bool = False

        # compare the inverse and the transpose of the matrix
        if not np.allclose(np.linalg.inv(transform[:3,:3]), transform[:3,:3].T) and np.linalg.det(transform[:3,:3]) != 1:
            print(f'{i} - Not Homogenous')
            print(transform.T)
            print(np.linalg.inv(transform))

            homo_bool = False

    print("The transforms are Homogeneous" if homo_bool else "Some or all of the transforms are not Homogeneous")

def correct_lidar_odometry(lidar_odom_raw, init_heading):

    init_heading_ENU = utils.convCompassENU(init_heading)
    
    rot_mat_enu = R.from_euler("z", init_heading_ENU, degrees=True).as_matrix()

    # Correcting the odometry from sensor frame to ENU frame:
    lidar_odom_rot = (rot_mat_enu @ lidar_odom_raw[:,:3].T).T

    # Then we add an offset to the relative heading (in ENU frame):
    return np.column_stack((lidar_odom_rot, lidar_odom_raw[:, 5]))