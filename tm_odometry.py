import numpy as np
import utils
from pymap3d import geodetic2enu, enu2geodetic
import osmnx as ox
from PIL import Image
import matplotlib.pyplot as plt
from math import ceil
import cv2 as cv
from pathlib import Path
import open3d as o3d
from scipy.spatial.transform import Rotation as R
import hdbscan
from math import acos, pi
import sys


def createOSMapImage(params):
        """
        This function creates a gray-scale image of the OpenStreetMap map for a given bounding box.
        The map includes features: coastline, buildings, ferry terminals and piers. (On some occasions the coastline is not tagged, so islet is included as a safeguard).
        The map is white for features and their boundaries. The background is black.
        Additionally, we add the map pxwidth to the parameters as it isn't consistently (500x500) as we want.

        Input:
        params: Parameters-object

        Output:
        osm_map_pxwidth: int
        """

        # We unpack the parameters:
        center = params.osm_map_center
        osm_max_range = params.osm_map_max_range
        map_fig_path = params.osm_map_fig_path
        
        # We create a south-west and north-east bound for the map (in latlon coordinates):
        south_west_bound = enu2geodetic(-osm_max_range, -osm_max_range, 0, center[0], center[1], 0)
        north_east_bound = enu2geodetic(osm_max_range, osm_max_range, 0, center[0], center[1], 0)

        # We create a bounding box for the map (in latlon coordinates):
        bbox = (south_west_bound[0], south_west_bound[1], north_east_bound[0], north_east_bound[1])

        # We use different tags for coastline and other points of interest:
        tags = {'building': True,
                'amenity': 'ferry_terminal',
                'man_made': 'pier',
                'place' : 'islet'}

        tag_coastline = {'natural': 'coastline',
                        'bridge': 'yes'}

        # bounding box is (north, south, east, west)
        try:
            coastline = ox.features_from_bbox(bbox[2], bbox[0], bbox[3], bbox[1], tag_coastline)
        except ox._errors.InsufficientResponseError:
            return None
        try:
            poi = ox.features_from_bbox(bbox[2], bbox[0], bbox[3], bbox[1], tags)
        except ox._errors.InsufficientResponseError:
            return None

        # We create a figure, which will illustrate the features on the map for the given bounding box:
        # The map will have a black background and white features.
        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        fig.patch.set_facecolor('black')
        poi.boundary.plot(ax = ax, color='white')
        coastline.plot(ax = ax, color='white')
        ax.set_xlim(bbox[1], bbox[3])
        ax.set_ylim(bbox[0], bbox[2])
        ax.set_axis_off()
        plt.savefig(map_fig_path, bbox_inches='tight', pad_inches=0)
        plt.close()

        return np.array(Image.open(map_fig_path)).shape[0]


def scan2MapImg(source, heading, params, scan_map_path, map_type):
    """
    This function takes a point cloud and creates a 2D-map image from it in gray-scale.
    Similarly to createOSMapImage, the points are represented by white pixels and the background is black.
    Furthermore, the scan is rotated according to the heading given.
    It's also scaled to fit the map.

    Input:
    source: o3d.geometry.PointCloud
    heading: float (degrees)
    params: Parameters-object
    scan_map_path: str
    map_type: str

    Output:
    scan_pixel_origin: tuple (int, int)
    """

    # We unpack the parameters:
    if map_type == 'osm':
        map_pxwidth = params.osm_map_pxwidth
        curr_max_range = params.osm_map_max_range
    elif map_type == 'enc':
        map_pxwidth = params.enc_map_pxwidth
        curr_max_range = params.enc_map_max_range
    
    voxel_size = params.voxel_size

    # We filter the scan :
    source_filtered = utils.filterScan(source, params)
    if source_filtered == None:
        print("Scan is empty")
        return None

    # If the point cloud is too small we skip it:
    # Less than HDBSCAN cluster_min_size points.
    if len(source_filtered.points) <= 5:
        print("Scan is too small")
        return None

    source_filtered, labels = clusterPointCloud(np.array(source_filtered.points), params)

    if source_filtered is None:
        print("Scan is empty (2)")
        return None
    
    if map_type == 'enc':
        source_filtered, labels = removeOccludedClusters(source_filtered, labels, params)
    
    curr_map = np.asarray(source_filtered)
    rotate_map = R.as_matrix(R.from_euler('z', -heading, degrees=True))
    curr_map = np.array([rotate_map @ point for point in curr_map])

    # We calculate the conversion from point (m) to pixel (px):
    dx_px = 2*curr_max_range/map_pxwidth

    if len(curr_map) == 0:
        print("Scan is empty")
        return None

    lidar_max_range = params.lidar_max_range
    # We create a map array to draw the points on:
    min_x = min(np.min(curr_map[:,0]), -lidar_max_range)
    max_x = max(np.max(curr_map[:,0]), lidar_max_range)
    min_y = min(np.min(curr_map[:,1]), -lidar_max_range)
    max_y = max(np.max(curr_map[:,1]), lidar_max_range)
    scan_map_img = np.zeros((ceil(np.abs(max_x - min_x)/voxel_size/dx_px), ceil(np.abs(max_y - min_y)/voxel_size/dx_px)))

    scan_pixel_origin = (int(np.abs(min_x)/voxel_size/dx_px), int(np.abs(max_y)/voxel_size/dx_px))
    # scan_map_img[template_origin_pixel[0], template_origin_pixel[1]] = 255 # Drawing origin pixel white
    # print(f' Template origin pixel: {scan_pixel_origin}')

    for point in curr_map[:,:2]:
            x = int((point[0] - min_x)/voxel_size/dx_px)
            y = int((point[1] - min_y)/voxel_size/dx_px)
            scan_map_img[x, y] = 255

    scan_map_img = scan_map_img.astype(np.uint8)
    # print(f' Template dimensions: {scan_map_img.shape}')
    im = Image.fromarray(scan_map_img)
    im.save(scan_map_path)

    return scan_pixel_origin




def matchingMap2Scan(map_fig_path, scan_map_path, scan_pixel_origin, method, map_type, plotting=False):
    """
    This function reads a map image and a scan image and performs template matching.
    The template matching is performed using the methods specified in the methods list.
    The function returns the results and the location of the best match (the best match is the upper-left corner of the template).

    Input:
    map_fig_path: str
    scan_map_path: str
    scan_pixel_origin: tuple (int, int)
    methods: list of strings
    map_type: str
    plotting: bool

    Output:
    results: list of floats
    results_loc: list of tuples (int, int)
    """
    
    
    # We read the image:
    img = cv.imread(map_fig_path, cv.IMREAD_COLOR)
    assert img is not None, "file could not be read, check with os.path.exists()"
    if map_type == 'enc':
        # We convert the ENC from blue, red and black to white (coastline) and black (everything else):
        blue_pixels = (img[:, :, 2] == 255)  # Check if the blue channel is equal to 255
        img[~blue_pixels] = [0, 0, 0]  # Set non-blue pixels to black
        img[blue_pixels] = [255, 255, 255]  # Set blue pixels to white

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img2 = img.copy()

    # We read the template (scan map):
    template = cv.imread(scan_map_path, cv.IMREAD_GRAYSCALE)
    assert template is not None, "file could not be read, check with os.path.exists()"
    w, h = template.shape[::-1]

    if plotting == True:
        plt.subplot(111),plt.imshow(template,cmap = 'gray')
        plt.title(f"example scan as template \n for {map_type.upper()}"), plt.xticks([]), plt.yticks([])
        plt.show()

    # result = 0
    # result_loc = (0, 0)

    img = img2.copy()
    # Apply template Matching
    res = cv.matchTemplate(img, template, eval(method))
    _, max_val, _, max_loc = cv.minMaxLoc(res)


    top_left = max_loc
    result = max_val
    result_loc = max_loc

    if plotting == True:
        bottom_right = (top_left[0] + w, top_left[1] + h)

        cv.rectangle(img, top_left, bottom_right, 255, 2)
        
        # A rectangle at the detected spot (corresponding to the origin):
        # Be aware that the pixel coordinates are inverted here (x,y) -> (y,x):
        res_top_left = (top_left[0] + scan_pixel_origin[1], top_left[1] + scan_pixel_origin[0])
        res_bottom_right = (res_top_left[0] + 5, res_top_left[1] + 5)
        cv.rectangle(img, res_top_left, res_bottom_right, 230, 5)

        plt.subplot(121),plt.imshow(res,cmap = 'gray')
        plt.title('Matching result'), plt.xticks([]), plt.yticks([])

        plt.subplot(122),plt.imshow(img,cmap = 'gray')
        plt.title('Detected area and point'), plt.xticks([]), plt.yticks([])

        plt.suptitle(f"example matching result and detected area \n for {map_type.upper()} with {method}")

        plt.tight_layout() # This makes sure the title doesn't overlap with the images

        plt.show()

    return result, result_loc

def computePositionEstimateMatching(matching_results, matching_results_loc, scan_pixel_origin, params, map_type='osm'):
    """
    This function computes the position estimate in latitude and longitude from the template matching results.


    Input:
    matching_results: list of floats
    matching_results_loc: list of tuples (int, int)
    scan_pixel_origin: tuple (int, int)
    params: Parameters-object
    map_type: str

    Output:
    pos_estimate_latlon: tuple (float, float)
    """

    # We unpack the necessary parameters:
    if map_type == 'osm':
        curr_max_range = params.osm_map_max_range
        map_center = params.osm_map_center
        map_pxwidth = params.osm_map_pxwidth

    elif map_type == 'enc':
        curr_max_range = params.enc_map_max_range
        map_center = params.enc_map_center
        map_pxwidth = params.enc_map_pxwidth

    dx_px = 2*curr_max_range/map_pxwidth

    # We calculate the conversion from point (m) to pixel (px):
    # Be aware that the pixel coordinates are inverted here (x,y) -> (y,x):
    pos_pixel = int(matching_results_loc[0] + scan_pixel_origin[0]), int(matching_results_loc[1] + scan_pixel_origin[1])

    # We calculate a relative position change from the map_center:
    pos_m = dx_px * (map_pxwidth/2  - np.asarray(pos_pixel))

    pos_estimate_latlon = enu2geodetic(pos_m[0], pos_m[1], 0, map_center[0], map_center[1], 0)

    return pos_estimate_latlon



def clusterPointCloud(points, params, reg=False):
    """
    Function to cluster a point cloud using HDBSCAN.
    It also removes noise and clusters that are too small. Which is used for registration.

    Input:
    points: (N x 3) numpy array OR o3d.geometry.PointCloud
    params: Parameters-object
    reg: bool

    Output:
    points: (M x 3) numpy array
    labels: (M x 1) numpy array

    if reg=True:
    pcl: o3d.geometry.PointCloud
    labels: (M x 1) numpy array
    """
    # We unpack the parameters:
    cluster_min_size = params.cluster_min_size

    if reg:
        points = np.asarray(points.points)

    # We perform clustering using HDBSCAN:
    HDBSCAN_clustering = hdbscan.HDBSCAN(min_cluster_size=cluster_min_size).fit(points)

    # We remove clusters that are labelled as noise:
    labels = HDBSCAN_clustering.labels_
    points = points[labels != -1]
    labels = labels[labels != -1]

    # We remove the clusters that are too small:
    unique, counts = np.unique(labels, return_counts=True)
    counts_mask = counts > cluster_min_size
    points = np.array([x for x, y in zip(points, labels) if counts_mask[y]])
    labels = np.array([y for y in labels if counts_mask[y]])

    if len(points) == 0:
        return None, None

    if reg:
        pcl = o3d.geometry.PointCloud()
        pcl.points = o3d.utility.Vector3dVector(points)
        return pcl, labels 

    return points, labels



def removeOccludedClusters(points, labels, params):
    """
    This function calculates the angle between the centroids of the clusters.
    If the angle is below the threshold, the cluster that is further away from the origin is removed.

    Input:
    points: (N x 3) numpy array
    labels: (N x 1) numpy array
    occlusion_angle_threshold: float

    Output:
    points: (M x 3) numpy array
    labels: (M x 1) numpy array
    """

    # We unpack the parameters:
    occlusion_angle_threshold = params.occlusion_angle_threshold

    # We calculate the centroids for each cluster:
    centroids = np.array([np.mean(points[labels == label], axis=0) for label in np.unique(labels)])
    # print(centroids)

    for i, (centroid, label) in enumerate(zip(centroids, np.unique(labels))):
        #print(f"Centroid {i}: {centroid} with label {label}")
        centroid_norm = np.linalg.norm(centroid)
        # print(centroid_norm)
        for j, (other_centroid, other_label) in enumerate(zip(centroids, np.unique(labels))):
            if i != j:
                # print(f"Distance to centroid {j}: {np.linalg.norm(centroid - other_centroid)}")
                other_centroid_norm = np.linalg.norm(other_centroid)
                # print(other_centroid_norm)
                centroids_angle = acos(np.dot(centroid, other_centroid)/(centroid_norm * other_centroid_norm))/pi*180
                
                if centroids_angle < 2 * occlusion_angle_threshold:
                    # We calculate which centroid is further away from the origin:
                    if centroid_norm > other_centroid_norm:
                        # We remove the other centroid:
                        points = points[labels != label]
                        labels = labels[labels != label]

    return points, labels


def createENCImage(center, max_range, timestamp, pxwidth, img_path):
    """
    This function creates an ENC image for the shoreline, from the ENC data (charts).

    Requirements: Access to SeaChartAPI from ShippingLab repository
    """

    sys.path += ['./SeaChartAPI']

    map_path = "./charts/DK5KOEBH.000"

    from SeaChartAPI import Coords, Draw
    from SeaChartAPI.seamap import SeaChart, filter_radar_range
    
    # BGR format
    BUOY_COLOR = (0, 255, 0, 255)
    LAND_COLOR = (255, 0, 0, 255)
    SHORE_COLOR = (0, 0, 255, 255)
    WATER_COLOR = (0, 0, 0, 255)
    NAVLINE_COLOR = (0, 255, 255, 255)
    

    sc = SeaChart(map_path)
    radius = max_range

    bottom_left, top_right = filter_radar_range(*center, radius)
    map_filter = (bottom_left, top_right)
    sc.setfilter(map_filter)
    sc.setdbg(0)
    land = sc.get_land()
    buoys = sc.get_buoyes()
    center = Coords(*center, "imgcenter")
    mapviz = Draw(pxwidth, WATER_COLOR, center, max_range / 1000, debugval=False, saveval=False)
    mapviz.setfilter(map_filter)

    mapviz.land(land, LAND_COLOR, SHORE_COLOR)
    mapviz.buoyes(buoy_data=buoys, buoys_color=BUOY_COLOR)
    mapviz.draw_structures(structure_color=LAND_COLOR, edge_color=SHORE_COLOR)
    fname = str(img_path / f"ENC_{timestamp}_maxrange{max_range}_width_{pxwidth}.png")
    if not Path(fname).exists():
        mapviz.show(show=False, save=True, name=fname)

        these_vals = get_class_bev(fname, SHORE_COLOR)
        cv.imwrite(fname, these_vals * 255)
    return fname


def get_class_bev(bev_img: str, class_int: tuple):
    """
    Auxiliary function for createENCImage.

    Input:
    bev_img: str
    class_int: tuple

    Output:
    "empty_array": (N x M) numpy array
    """
    img = cv.imread(bev_img, cv.IMREAD_UNCHANGED)
    where_to_fill = np.where(np.all(img == class_int, axis=-1))
    empty_array = np.zeros_like(img)
    empty_array[where_to_fill] = (1, 1, 1, 1)
    return empty_array.astype(int)[..., 0]




def tm_odometry_pipeline(map_type, pcd_paths, lidar_df, gps_df, heading_df, enc_df, params):
    """
    This function computes an odometry estimate using template matching on LiDAR scans with maps (ENC/OSM).

    Input:
    map_type: str
    lidar_df: DataFrame
    gps_df: DataFrame
    enc_df: DataFrame
    params: Parameters-object

    Output:
    pos_estimates_enu: np.ndarray
    """

    enc_paths = np.asarray(enc_df['path'].values)
    enc_paths = np.asarray([Path.home() / Path(p).relative_to("/home/jovyan") for p in enc_paths])

    map_times = np.array(gps_df['timestamp'].values)
    pos_estimate_enu = np.zeros((len(map_times), 3))
    pos_estimate_latlon = np.zeros((len(map_times), 3))

    method = 'cv.TM_CCOEFF_NORMED'

    for k, time in enumerate(map_times):
        curr_time = time


        idx_pcd = utils.closestIdx(np.array(lidar_df['timestamp'].values), curr_time)
        idx_heading = utils.closestIdx(np.array(heading_df['timestamp'].values), curr_time)
        idx_gps = utils.closestIdx(np.array(gps_df['timestamp'].values), curr_time)
        idx_enc = utils.closestIdx(np.array(enc_df['timestamp'].values), curr_time)

        position_latlon = gps_df[['latitude', 'longitude']].values[idx_gps]
        position_enu = gps_df[['East', 'North']].values[idx_gps]


        gps_center = gps_df[['latitude', 'longitude']].values[idx_gps]
        map_max_range = enc_df['enc_max_range'].values[idx_enc]
        osm_map_fig_path = "./map_imgs/osm_map.png"

        enc_map_center = enc_df[['latitude', 'longitude']].values[idx_enc]
        enc_map_fig_path = str(enc_paths[idx_enc])
        enc_map_pxwidth = enc_df['pxwidth'].values[idx_enc]
        
        params.add_parameters(osm_map_center=gps_center, osm_map_max_range=map_max_range, osm_map_fig_path=osm_map_fig_path)
        params.add_parameters(enc_map_center=enc_map_center, enc_map_max_range=map_max_range, 
                        enc_map_fig_path=enc_map_fig_path, enc_map_pxwidth=enc_map_pxwidth)
        
        if map_type == "osm":
            map_pxwidth = createOSMapImage(params)
            if map_pxwidth is None:
                    continue
            
            params.add_parameters(osm_map_pxwidth = map_pxwidth)

        
        scan_map_path = "./map_imgs/scan_map.png"
        source = np.asarray(o3d.io.read_point_cloud(str(pcd_paths[idx_pcd])).points)
        scan_pixel_origin = scan2MapImg(source, heading_df['heading vessel'].values[idx_heading], params, scan_map_path, map_type=map_type)
        if scan_pixel_origin is None:
                continue
        
        
        if map_type == "osm":
                map_fig_path = params.osm_map_fig_path
        elif map_type == "enc":
                map_fig_path = params.enc_map_fig_path
        results, results_loc = matchingMap2Scan(map_fig_path, scan_map_path, scan_pixel_origin, method, map_type, plotting=False)

        
        
        pos_estimate_latlon[k] = computePositionEstimateMatching(results, results_loc, scan_pixel_origin, params, map_type)


        pos_estimate_enu[k] = np.asarray(geodetic2enu(pos_estimate_latlon[k, 0], pos_estimate_latlon[k, 1], pos_estimate_latlon[k, 2], gps_df['latitude'].values[0], gps_df['longitude'].values[0], 0))

        # print(f'Position estimate at LiDAR scan timestamp {lidar_timestamps_relative[idx_pcd]}: {pos_estimate_latlon}')
        # print(f'Actual (closest gps) position at timestamp {gps_timestamps_relative[idx_gps]}: {position_latlon}')
        # print(f'Error in position estimate: {np.linalg.norm(pos_estimate_enu[:2] - position_enu)}')
    
    return np.column_stack((map_times, pos_estimate_enu))