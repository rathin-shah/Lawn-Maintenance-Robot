import numpy as np
import transforms3d as tf


class InverseKinematics(object):
    STEERING_NAME = 'steering'
    DRIVING_NAME = 'driving'
    MIN_SPEED = 0.01

    def __init__(self, config):
        self.rover_config = config
        self.r_sv = {}
        self.r_cs = {}
        for joint_name in config['joint_names']:
            wheel_name, joint_type = joint_name.split('/')
            if joint_type == 'driving':
                self.r_cs[wheel_name] = np.array([
                    config['joint_x'][joint_name],
                    config['joint_y'][joint_name],
                    0.0
                ])
            elif joint_type == 'steering':
                self.r_sv[wheel_name] = np.array([
                    config['joint_x'][joint_name],
                    config['joint_y'][joint_name],
                    0.0
                ])

    def _clamp_velocity(self, velocity):
        """
        Clamp velocity to 1 cm/s
        Sending a very small velocity will get clamped here, but the steering
        will still go through.

        We don't want a lower resolution than 1 cm/s anyways

        return clamped velocity
        """
        return velocity / \
            self.rover_config['wheel_radius'] if (
                abs(velocity) > self.MIN_SPEED) else 0.0

    def _append_position(self, ang, positions, control_types):
        """
        Wrapper to simply append angle value and position control types to arrays
        """
        positions.append(ang)
        control_types.append('position')

    def _append_velocity(self, vel, velocities, control_types):
        """
        Wrapper to simply append velocity value and velocity control types to arrays
        """
        velocities.append(vel)
        control_types.append('velocity')

    def _calc_steering(self, v_swv):
        raw_ang = -np.arctan2(v_swv[1], v_swv[0])
        return (raw_ang)   ##need to clamp
  
    def apply(self, vx, vy, wz, steer_vels=None):
        values = []
        control_types = []
        targets = []

        v_vwv = np.array([vx, vy, 0.0])
        w_vwv = np.array([0.0, 0.0, wz])

        for wheel_name, r_sv in self.r_sv.items():
            steer_vel = 0.0
            if steer_vels is not None:
                steer_vel = steer_vels[wheel_name]
            else:
                steer_vel = 0.0
            v_swv = v_vwv + np.cross(w_vwv, r_sv)

            # Calculate steering angle for each wheel
            targets.append('/'.join([wheel_name, self.STEERING_NAME]))
            steering_ang = self._calc_steering(v_swv)
            dirZ = -1.0 if self.rover_config['steer_Z_down'] else 1.0
            self._append_position(dirZ * steering_ang, values, control_types)

            # Calculate driving angular rate for each wheel
            targets.append('/'.join([wheel_name, self.DRIVING_NAME]))
            R_w_B = np.array([[np.cos(-steering_ang), np.sin(-steering_ang), 0.0],
                             [np.sin(-steering_ang), -np.cos(-steering_ang), 0.0], [0.0, 0.0, -1.0]])

            R_B_s = R_w_B.T
            v_b_off_contrib = np.cross(
                np.array([0.0, 0.0, wz]), R_B_s @ self.r_cs[wheel_name])
            v_st_off_contrib = np.cross(
                np.array([0.0, 0.0, steer_vel]), R_B_s @ self.r_cs[wheel_name])

            v_wvw = R_w_B @ (v_swv + v_b_off_contrib + v_st_off_contrib)
            driving_vel = self._clamp_velocity(v_wvw[0])
            self._append_velocity(driving_vel, values, control_types)
        return targets, values, control_types