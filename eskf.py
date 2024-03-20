import numpy as np
from parameters import loadParameters
from scipy.spatial.transform import Rotation as R
import utils

params = loadParameters()

class ESKF():
    '''
    Class providing the implementation of the Error-State Kalman Filter (ESKF).
    '''

    def __init__(self, initial_heading, length_imu):
        '''
        Initializes the ESKF with the given initial heading and the amount of IMU measurements.

        Input:
            initial_heading (float): Initial heading w.r.t True North in degrees.
            length_imu (int): Amount of IMU measurements.
        
        Implicit input:
            params (Parameters): Parameters object.
        '''

        self.F_i = params.F_i
        self.P = np.zeros((length_imu,params.n_states, params.n_states))
        self.x_hat = np.zeros((length_imu, params.n_states))
        
        # Initialize the state and covariance:
        self.P[0] = params.P_init
        self.x_hat[0] = np.array([0, 0, 0, 0, 0, 0, initial_heading, params.acc_b_0, params.acc_b_0, params.acc_b_0,
                               params.gyr_b_0, 0, 0, 0])

    def propagate(self, k, dt, accel, gyro):
        '''
        Propagates the state and covariance using the IMU input.

        Input:
            dt (float): Time step.
            accel (np.ndarray): Accelerometer measurement.
            gyro (np.ndarray): Gyroscope measurement.

        Implicit input:
            params (Parameters): Parameters object.
        '''

        # IMU measurements:
        accel = accel - self.x_hat[k - 1, 7:10]
        gyro = gyro - self.x_hat[k - 1, 10:11]
        R_temp = R.from_euler('z', self.x_hat[k - 1, 6], degrees=True).as_matrix()

        # Using the kinematic model:
        # Update position estimate:
        self.x_hat[k, 0:3] = self.x_hat[k - 1, 0:3] + self.x_hat[k - 1, 3:6] * dt + 0.5 * (R_temp @ accel + params.g) * (dt**2)

        # Update velocity estimate:
        self.x_hat[k, 3:6] = self.x_hat[k - 1, 3:6] + dt * (R_temp @ accel + params.g)

        # Update heading estimate:
        self.x_hat[k, 6] = self.x_hat[k - 1, 6] + dt * (gyro/np.pi * 180)

        # Update biases:
        self.x_hat[k, 7:10] = self.x_hat[k - 1, 7:10]
        self.x_hat[k, 10:11] = self.x_hat[k - 1, 10:11]
        self.x_hat[k, 11:14] = np.exp(-params.lambda_l*dt)*self.x_hat[k - 1, 11:14]


        # 1.1 Linearize the motion model and compute Jacobians (see also report for details on derivation):
        F_x = np.eye(params.n_states)
        # Row 1
        F_x[0:3, 3:6] = np.eye(3) * dt
        # Row 2
        F_x[3:6, 7:10] = -R_temp * dt
        u = np.array([0, 0, 1]).reshape(3, 1)
        F_x[3:6, 6] = (-R_temp * utils.skew_symmetric(accel) @ u * dt).reshape(3)
        # Row 3
        F_x[6, 10] = -dt
        # Row 6
        F_x[11:14, 11:14] = np.eye(3) * np.exp(-params.lambda_l*dt)

        # 2. Propagate uncertainty
        Q = np.eye(self.F_i.shape[1])
        Q[0:3, 0:3] *= params.sigma_vn**2 * dt**2 # Velocity impulse
        Q[3:4, 3:4] *= params.sigma_thetan**2 * dt**2 # Angular/heading impulse
        Q[4:7, 4:7] *= params.sigma_aw**2 * dt # Acceleration bias impulse
        Q[7:8, 7:8] *= params.sigma_ww**2 * dt # Gyro bias impulse
        Q[8:11, 8:11] *= params.sigma_ln**2 * dt # Lidar Odometry bias impulse


        self.P[k] = F_x @ self.P[k - 1] @ F_x.T + self.F_i @ Q @ self.F_i.T
    
    def get_measurement_matrix(self, measurement_type):
        '''
        Returns the measurement matrix and the predicted measurement vector.

        Args:
            measurement_type (str): Measurement type.

        Returns:
            np.ndarray: Measurement matrix.
        
        '''
        if measurement_type == "gps_vel":
            return params.H_pv
        
        elif measurement_type == "gps":
            return params.H_gps

        elif measurement_type == "heading":
            return params.H_head

        elif measurement_type == "lidar":
            return params.H_lidar

        elif measurement_type == "tm":
            return params.H_tm


    def measurement(self, k, measurement_type, y_m, y_noise):
        '''
        Updates the state and covariance using the given measurement.

        Inputs:
            k (int): Current time step.
            y_hat (np.ndarray): Predicted measurement vector.
            measurement_type (str): Measurement type.
            y_m (np.ndarray): Measurement vector (sensor).
            y_noise -> measurement_noise (np.ndarray): Measurement noise covariance matrix.

        '''
        H = self.get_measurement_matrix(measurement_type)
        y_hat = self.predict_measurement(k, measurement_type)
        y_noise = y_noise**2

        # 3.1 Compute Kalman Gain
        try:
            K = self.P[k] @ H.T @ np.linalg.inv(H @ self.P[k] @ H.T + y_noise)
            
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                raise "A singular matrix "
            
        # 3.2 Compute error state
        error_state = y_m - y_hat
        
        # 3.3 Correct predicted state
        x_error = K @ error_state
        self.x_hat[k] = self.x_hat[k] + x_error

        # 3.4 Compute corrected covariance (Joseph Formula):
        self.P[k] = (np.eye(params.n_states) - K @ H) @ self.P[k] @ (np.eye(params.n_states) - K @ H).T + K @ y_noise @ K.T

    def predict_measurement(self, k, measurement_type):
        '''
        Predicts the measurement vector.

        Args:
            measurement_type (str): Measurement type.

        Returns:
            np.ndarray: Predicted measurement vector.
        '''
        H = self.get_measurement_matrix(measurement_type)
        return H @ self.x_hat[k]
