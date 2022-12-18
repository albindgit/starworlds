from obstacles import Obstacle, Frame
from utils import is_cw, is_ccw, is_collinear, tic, toc
import shapely
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Polygon(Obstacle):

    def __init__(self, polygon, **kwargs):
        super().__init__(**kwargs)
        self._polygon = shapely.geometry.Polygon(polygon)
        self._polygon = shapely.ops.orient(self._polygon)
        self._pol_bounds = self._polygon.bounds
        self._compute_global_polygon_representation()
        self.vertices = np.array(self._polygon.exterior.coords[:-1])
        self.circular_vertices = np.array(self._polygon.exterior.coords)

    def init_plot(self, ax=None, show_name=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        if "fc" not in kwargs and "facecolor" not in kwargs:
            kwargs["fc"] = 'lightgrey'
        if 'show_reference' in kwargs:
            del kwargs['show_reference']
        line_handles = []
        # Boundary
        line_handles += [patches.Polygon(np.random.rand(3, 2), **kwargs)]
        ax.add_patch(line_handles[-1])
        # Name
        line_handles += [ax.text(0, 0, self._name)] if show_name else [None]
        return line_handles, ax

    def extreme_points(self, frame=Frame.GLOBAL):
        vertices = np.asarray(self.polygon(frame).exterior.coords)[:-1, :]
        return [vertices[i] for i in range(vertices.shape[0])]

    def update_plot(self, line_handles, frame=Frame.GLOBAL):
        polygon = self.polygon(frame)
        boundary = np.vstack((polygon.exterior.xy[0], polygon.exterior.xy[1])).T
        line_handles[0].set_xy(boundary)
        if line_handles[1] is not None:
            line_handles[1].set_position(self.pos(frame))

    def draw(self, frame=Frame.GLOBAL, **kwargs):
        line_handles, ax = self.init_plot(**kwargs)
        self.update_plot(line_handles, frame)
        return line_handles, ax

    def dilated_obstacle(self, padding, id="new", name=None):
        cp = self.copy(id, name)
        cp._polygon = cp._polygon.buffer(padding, cap_style=1, join_style=1)
        cp._pol_bounds = cp._polygon.bounds
        cp.vertices = np.array(cp._polygon.exterior.coords[:-1])
        cp.circular_vertices = np.array(cp._polygon.exterior.coords)
        cp._polygon_global_pose = None
        cp._polygon_global = None
        return cp

    def point_location(self, x, input_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        xmin, ymin, xmax, ymax = self._pol_bounds
        if not (xmin < x_obstacle[0] < xmax and ymin < x_obstacle[1] < ymax):
            return 1
        x_sh = shapely.geometry.Point(x_obstacle)
        if self._polygon.contains(x_sh):
            return -1
        if self._polygon.exterior.contains(x_sh):
            return 0
        return 1

    def line_intersection(self, line, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        l0_obstacle = self.transform(line[0], input_frame, Frame.OBSTACLE)
        l1_obstacle = self.transform(line[1], input_frame, Frame.OBSTACLE)
        intersection_points_shapely = shapely.geometry.LineString([l0_obstacle, l1_obstacle]).intersection(self._polygon.exterior)
        if intersection_points_shapely.is_empty:
            return []
        if intersection_points_shapely.geom_type == 'Point':
            intersection_points_obstacle = [np.array([intersection_points_shapely.x, intersection_points_shapely.y])]
        elif intersection_points_shapely.geom_type == 'MultiPoint':
            intersection_points_obstacle = [np.array([p.x, p.y]) for p in intersection_points_shapely.geoms]
        elif intersection_points_shapely.geom_type == 'LineString':
            intersection_points_obstacle = [np.array([ip[0], ip[1]]) for ip in intersection_points_shapely.coords]
        elif intersection_points_shapely.geom_type == 'MultiLineString':
            intersection_points_obstacle = [np.array([ip[0], ip[1]]) for line in intersection_points_shapely.geoms for ip in line.coords]
        else:
            print(intersection_points_shapely)
        return [self.transform(ip, Frame.OBSTACLE, output_frame) for ip in intersection_points_obstacle]

    def tangent_points(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):

        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        t0 = tic()
        phi = np.arctan2(self.circular_vertices[:, 1] - x_obstacle[1], self.circular_vertices[:, 0] - x_obstacle[0])
        phi[phi < 0] += 2 * np.pi
        t1 = toc(t0)
        t0 = tic()
        phi_diff = np.diff(phi)
        t2 = toc(t0)
        t0 = tic()
        phi_decrease_idcs = phi_diff > np.pi
        phi_increase_idcs = phi_diff < -np.pi
        t3 = toc(t0)
        t0 = tic()
        phi_decrease_idcs = np.flatnonzero(phi_decrease_idcs)
        phi_increase_idcs = np.flatnonzero(phi_increase_idcs)
        for i in phi_decrease_idcs:
            phi[i+1:] -= 2*np.pi
        for i in phi_increase_idcs:
            phi[i+1:] += 2*np.pi
        t4 = toc(t0)

        t0 = tic()

        i_min, i_max = np.argmin(phi), np.argmax(phi)

        if abs(phi[0] - phi[-1]) > 0.00001:
            # Interior point
            return []
        if (phi[i_max] - phi[i_min]) >= 2*np.pi:
            # Blocked exterior point
            return []
        t5 = toc(t0)

        t0 = tic()
        tp1_obstacle = self.circular_vertices[i_max]
        tp2_obstacle = self.circular_vertices[i_min]

        tp1 = self.transform(tp1_obstacle, Frame.OBSTACLE, output_frame)
        tp2 = self.transform(tp2_obstacle, Frame.OBSTACLE, output_frame)

        tend = toc(t0)
        # print(sum([t1*100,t2*100,t3*100,t4*100,t5*100,tend*100*0]))
        return [tp1, tp2]

    def area(self):
        return self._polygon.area

    # ------------ Private methods ------------ #
    def _check_convexity(self):
        v = np.asarray(self._polygon.exterior.coords)[:-1, :]
        i = 0
        N = v.shape[0]
        # Make sure first vertice is not collinear
        while is_collinear(v[i-1, :], v[i, :], v[(i+1) % N, :]):
            i += 1
            if i > N:
                raise RuntimeError("Bad polygon shape. All vertices collinear")
        # All vertices must be either cw or ccw when iterating through for convexity
        if is_cw(v[i-1, :], v[i, :], v[i+1, :]):
            self._is_convex = not any([is_ccw(v[j-1, :], v[j, :], v[(j+1) % N, :]) for j in range(v.shape[0])])
        else:
            self._is_convex = not any([is_cw(v[j-1, :], v[j, :], v[(j+1) % N, :]) for j in range(v.shape[0])])

    # Not needed
    def _compute_polygon_representation(self):
        pass
