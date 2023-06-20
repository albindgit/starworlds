import numpy as np
from abc import ABC, abstractmethod


class MotionModel(ABC):

    def __init__(self, pos=None, rot=0):
        self._pos = np.array([0., 0.]) if pos is None else np.array(pos, dtype='float64')
        self._rot = rot
        self._t = 0.

    def move(self, obs_self, dt):
        xy_vel = np.array(self.lin_vel())
        rot_vel = self.rot_vel()
        prev_pos, prev_rot = self._pos.copy(), self._rot
        self._pos += xy_vel * dt
        self._rot += rot_vel * dt
        self._t += dt

    def set_pos(self, pos):
        self._pos = pos

    def set_rot(self, rot):
        self._rot = rot

    def pos(self):
        return self._pos

    def rot(self):
        return self._rot

    @abstractmethod
    def lin_vel(self):
        pass

    @abstractmethod
    def rot_vel(self):
        pass


class Static(MotionModel):

    def lin_vel(self):
        return np.zeros(2)

    def rot_vel(self):
        return 0.


class SinusVelocity(MotionModel):

    # If cartesian_coords: (x1,x2) vel are Cartesian (x,y) vel.
    # Else:                (x1,x2) vel are linear and rotational (lin,ang) vel.
    def __init__(self, pos=None, rot=0, cartesian_coords=True, x1_mag=0., x1_period=0, x2_mag=0., x2_period=0):
        super().__init__(pos, rot)
        self.cartesian_coords = cartesian_coords
        self.x1_vel_mag = x1_mag
        self.x1_vel_freq = 2 * np.pi / x1_period if x1_period else 0
        self.x2_vel_mag = x2_mag
        self.x2_vel_freq = 2 * np.pi / x2_period if x2_period else 0

    def lin_vel(self):
        if self.cartesian_coords:
            return np.array([self.x1_vel_mag * np.cos(self.x1_vel_freq * self._t),
                             self.x2_vel_mag * np.cos(self.x2_vel_freq * self._t)])
        else:
            rot_mat = np.array([[np.cos(self._rot), -np.sin(self._rot)], [np.sin(self._rot), np.cos(self._rot)]])
            return rot_mat.dot([self.x1_vel_mag * np.cos(self.x1_vel_freq * self._t), 0])

    def rot_vel(self):
        if self.cartesian_coords:
            return 0.
        else:
            return np.array(self.x2_vel_mag * np.cos(self.x2_vel_freq * self._t))


class Interval(MotionModel):

    def __init__(self, init_pos, time_pos):
        self.pos_point = np.array([init_pos] + [p for _, p in time_pos])
        self.time_point = np.cumsum([0] + [t for t, _ in time_pos])
        super().__init__(init_pos, 0)

    def lin_vel(self):
        if self._t > self.time_point[-1]:
            vel_norm = np.linalg.norm((self.pos_point[-1] - self.pos_point[-2]) / (self.time_point[-1] - self.time_point[-2]))
            dir = self.pos_point[-1] - self.pos()
            if np.linalg.norm(dir) > vel_norm:
                dir /= np.linalg.norm(dir)
            return vel_norm * dir
        idx = np.argmax(self._t < self.time_point)
        return (self.pos_point[idx] - self.pos_point[idx-1]) / (self.time_point[idx] - self.time_point[idx-1])

    def rot_vel(self):
        return 0.


class Waypoints(MotionModel):

    def __init__(self, init_pos, waypoints, vel, wp_thresh=0.2):
        self.waypoints = np.array([init_pos] + waypoints)
        self.vel = vel
        self._wp_idx = 0
        self.wp_thresh = wp_thresh
        super().__init__(init_pos, 0)

    def lin_vel(self):
        if not self._wp_idx < len(self.waypoints):
            return np.zeros(2)
        dir = self.waypoints[self._wp_idx] - self.pos()
        wp_dist = np.linalg.norm(dir)
        if wp_dist < self.wp_thresh:
            self._wp_idx += 1
            return self.lin_vel()
        dir /= wp_dist
        return self.vel * dir

    def rot_vel(self):
        return 0.
