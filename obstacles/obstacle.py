from abc import ABC, abstractmethod
import numpy as np
import shapely
from shapely import affinity as sh_affinity
from utils import affine_transform
from copy import deepcopy
from enum import Enum


class Frame(Enum):
    GLOBAL = 1
    OBSTACLE = 2

    class InvalidFrameError(Exception):
        pass


class Obstacle(ABC):
    """ Abstract base class of obstacles
    """
    id_counter = 0

    # obs_id <0: temp object, obs_id=0: new object, obs_id>0: existing object with id #obs_id
    def __init__(self, motion_model=None, is_convex=None, is_starshaped=None, id='new', name=None, compute_polygon=False):
        self._id = None
        self._is_convex = is_convex
        self._is_starshaped = is_starshaped
        # Pose of local frame in global frame
        self._motion_model = motion_model  # if motion_model is not None else mm.Static([0., 0.], 0.)
        # Initialize id and name
        self._set_id_name(id, name)
        self._polygon = None  # Polygon in obstacle frame
        self._polygon_global = None  # Polygon in global frame if static obstacle
        self._polygon_global_pose = None  # Obstacle pose corresponding to global polygon
        if compute_polygon:
            self._compute_global_polygon_representation()

    def __str__(self): return self._name

    def copy(self, id='temporary', name=None):
        ob = deepcopy(self)
        if not (id == 'duplicate' or id == 'd'):
            ob._set_id_name(id, name)
        return ob

    def pos(self, output_frame=Frame.GLOBAL):
        if output_frame == Frame.OBSTACLE or self._motion_model is None:
            return np.zeros(2)
        if output_frame == Frame.GLOBAL:
            return self._motion_model.pos()

    def rot(self, output_frame=Frame.GLOBAL):
        if output_frame == Frame.OBSTACLE or self._motion_model is None:
            return 0.
        if output_frame == Frame.GLOBAL:
            return self._motion_model.rot()

    def interior_point(self, x, input_frame=Frame.GLOBAL):
        return True if self.point_location(x, input_frame) < 0 else False

    def exterior_point(self, x, input_frame=Frame.GLOBAL):
        return True if self.point_location(x, input_frame) > 0 else False

    def boundary_point(self, x, input_frame=Frame.GLOBAL):
        return np.isclose(self.point_location(x, input_frame), 0.)

    def move(self, dt):
        if self._motion_model is not None:
            self._motion_model.move(self, dt)

    def id(self): return self._id

    def polygon(self, output_frame=Frame.GLOBAL):
        if self._polygon is None:
            self._compute_polygon_representation()
            # self._compute_global_polygon_representation()
        if output_frame == Frame.OBSTACLE or self._motion_model is None:
            return self._polygon
        elif output_frame == Frame.GLOBAL:
            current_pose = [*self._motion_model.pos(), self._motion_model.rot()]
            if not current_pose == self._polygon_global_pose:
                self._polygon_global_pose = current_pose
                c, s = np.cos(current_pose[2]), np.sin(current_pose[2])
                trans_matrix = np.array([[c, -s, current_pose[0]], [s, c, current_pose[1]], [0, 0, 1]])
                affinity_matrix = [trans_matrix[0, 0], trans_matrix[0, 1], trans_matrix[1, 0], trans_matrix[1, 1], trans_matrix[0, 2], trans_matrix[1, 2]]
                self._polygon_global = sh_affinity.affine_transform(self._polygon, affinity_matrix)
            return self._polygon_global
        else:
            raise Frame.InvalidFrameError

    def intersects(self, other):
        return self.polygon().intersects(other.polygon())

    def transform(self, x, input_frame, output_frame):
        if input_frame == output_frame or self._motion_model is None:
            return x
        elif input_frame == Frame.OBSTACLE and output_frame == Frame.GLOBAL:
            return self._transform_obstacle2global(x)
        elif input_frame == Frame.GLOBAL and output_frame == Frame.OBSTACLE:
            return self._transform_global2obstacle(x)
        else:
            raise Frame.InvalidFrameError

    def rotate(self, x, input_frame, output_frame):
        if input_frame == output_frame or self._motion_model is None:
            return x
        elif input_frame == Frame.OBSTACLE and output_frame == Frame.GLOBAL:
            return self._rotate_obstacle2global(x)
        elif input_frame == Frame.GLOBAL and output_frame == Frame.OBSTACLE:
            return self._rotate_global2obstacle(x)
        else:
            raise Frame.InvalidFrameError

    def is_convex(self):
        # Check if convexity already has been computed
        if self._is_convex is None:
            self._check_convexity()
        return self._is_convex

    def is_starshaped(self):
        if self._is_starshaped is None:
            if self.is_convex():
                self._is_starshaped = True
            else:
                # TODO: Add check for starshapedness. Currently default to not starshaped
                self._is_starshaped = False
        return self._is_starshaped

    def set_motion_model(self, motion_model):
        self._motion_model = motion_model

    # ------------ Private methods ------------ #
    def _set_id_name(self, id, name=None):
        if id == 'new' or id == 'n':
            Obstacle.id_counter += 1
            self._id = Obstacle.id_counter
        elif id == 'temporary' or id == 'temp' or id == 't':
            self._id = None
        elif isinstance(id, int) and 0 < id <= Obstacle.id_counter:
            self._id = id
        else:
            print("Invalid id '" + str(id) + "' in set_id. Create temporary obstacle.")
            self._id = None
        self._name = name if name else str(self._id)

    def _rotate_obstacle2global(self, x_obstacle):
        rot = self._motion_model.rot()
        return affine_transform(x_obstacle, rotation=rot, translation=[0, 0])

    def _rotate_global2obstacle(self, x_global):
        rot = self._motion_model.rot()
        return affine_transform(x_global, rotation=rot, translation=[0, 0], inverse=True)

    def _transform_obstacle2global(self, x_obstacle):
        pos, rot = (self._motion_model.pos(), self._motion_model.rot())
        return affine_transform(x_obstacle, rotation=rot, translation=pos)

    def _transform_global2obstacle(self, x_global):
        pos, rot = (self._motion_model.pos(), self._motion_model.rot())
        return affine_transform(x_global, rotation=rot, translation=pos, inverse=True)

    def _compute_global_polygon_representation(self):
        self._compute_polygon_representation()
        if self._motion_model is None:
            self._polygon_global = self._polygon
        if self._motion_model.__class__.__name__ == 'Static':
            pos, rot = (self._motion_model.pos(), self._motion_model.rot())
            c, s = np.cos(rot), np.sin(rot)
            trans_matrix = np.array([[c, -s, pos[0]], [s, c, pos[1]], [0, 0, 1]])
            affinity_matrix = [trans_matrix[0, 0], trans_matrix[0, 1], trans_matrix[1, 0], trans_matrix[1, 1],
                               trans_matrix[0, 2], trans_matrix[1, 2]]
            self._polygon_global = shapely.affinity.affine_transform(self._polygon, affinity_matrix)

    # ------------ Abstract methods ------------ #
    @abstractmethod
    def draw(self):
        pass

    @abstractmethod
    def point_location(self, x, input_frame=Frame.GLOBAL):
        pass

    @abstractmethod
    def dilated_obstacle(self, padding, id="new", name=None):
        pass

    @abstractmethod
    def line_intersection(self, line, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        pass

    @abstractmethod
    def tangent_points(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        pass

    @abstractmethod
    def _check_convexity(self):
        pass

    @abstractmethod
    def _compute_polygon_representation(self):
        pass
