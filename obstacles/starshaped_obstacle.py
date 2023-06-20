from abc import abstractmethod
import numpy as np
from obstacles import Obstacle, Frame
from shapely import affinity as sh_affinity


class StarshapedObstacle(Obstacle):

    def __init__(self, xr, **kwargs):
        super().__init__(is_starshaped=True, **kwargs)
        self._xr = np.array(xr)  # Reference point in obstacle frame
        self._kernel = None
        self._kernel_global = None  # Kernel in global frame
        self._kernel_global_pose = None  # Obstacle pose corresponding to global kernel

    def xr(self, output_frame=Frame.GLOBAL):
        return self.transform(self._xr, Frame.OBSTACLE, output_frame)

    def set_xr(self, xr, input_frame=Frame.OBSTACLE, safe_set=False):
        new_xr = self.transform(xr, input_frame, Frame.OBSTACLE)
        if safe_set:
            k = self.kernel()
            if not k.exterior_point(new_xr, Frame.OBSTACLE):
                self._xr = new_xr
        else:
            self._xr = new_xr

    def reference_direction(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        dir = self.transform(x, input_frame, output_frame) - self.transform(self._xr, Frame.OBSTACLE, output_frame)
        if not np.any(dir):
            print("reference_direction for xr is not defined")
        return dir / np.linalg.norm(dir, axis=x.ndim - 1)

    # interior point: <1. exterior point: >1. boundary point: 1.
    def distance_function(self, x, input_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        dist_func = (np.linalg.norm(x_obstacle - self._xr, axis=x.ndim - 1) / (
            np.linalg.norm(self.boundary_mapping(x_obstacle, input_frame=Frame.OBSTACLE, output_frame=Frame.OBSTACLE)
                           - self._xr, axis=x.ndim - 1))) ** 2
        return dist_func

    # interior point: <0. exterior point: >0. boundary point: 0.
    def point_location(self, x, input_frame=Frame.GLOBAL):
        return np.sign(self.distance_function(x, input_frame)-1.)

    def kernel(self, output_frame=Frame.GLOBAL):
        if self._kernel is None:
            self._compute_kernel()
        if output_frame == Frame.OBSTACLE or self._motion_model is None:
            return self._kernel
        elif output_frame == Frame.GLOBAL:
            current_pose = [*self._motion_model.pos(), self._motion_model.rot()]
            if not current_pose == self._kernel_global_pose:
                self._kernel_global_pose = current_pose
                c, s = np.cos(current_pose[2]), np.sin(current_pose[2])
                trans_matrix = np.array([[c, -s, current_pose[0]], [s, c, current_pose[1]], [0, 0, 1]])
                affinity_matrix = [trans_matrix[0, 0], trans_matrix[0, 1], trans_matrix[1, 0], trans_matrix[1, 1], trans_matrix[0, 2], trans_matrix[1, 2]]
                self._kernel_global = sh_affinity.affine_transform(self._kernel, affinity_matrix)
            return self._kernel_global
        else:
            raise Frame.InvalidFrameError

    def vel_intertial_frame(self, x):
        if self._motion_model is None:
            return np.zeros(2)
        omega = self._motion_model.rot_vel()
        lin_vel_omega = np.cross(np.hstack(([0, 0], omega)), np.hstack((self.reference_direction(x), 0)))[:2]
        return self._motion_model.lin_vel() + lin_vel_omega

    # ------------ Abstract methods ------------ #
    @abstractmethod
    def normal(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL, x_is_boundary=False):
        pass

    @abstractmethod
    def boundary_mapping(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        pass

    # @abstractmethod
    # def signed_boundary_distance(self, x, input_frame=Frame.GLOBAL):
    #     pass

    @abstractmethod
    def _compute_kernel(self):
        pass
