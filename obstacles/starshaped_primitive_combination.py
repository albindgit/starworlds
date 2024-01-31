import shapely
import numpy as np
from obstacles import Frame, StarshapedObstacle, StarshapedPolygon
from utils import is_ccw, is_cw, draw_shapely_polygon
import matplotlib.pyplot as plt

# Note: Local == Global frame
class StarshapedPrimitiveCombination(StarshapedObstacle):

    def __init__(self, obstacle_cluster, hull_cluster, xr, **kwargs):
        self._obstacle_cluster = obstacle_cluster
        self._hull_cluster = hull_cluster
        super().__init__(xr=xr, **kwargs)
        self.vertices = None
        self.circular_vertices = None
        self.vertex_angles = None

    def obstacle_cluster(self):
        return self._obstacle_cluster

    def hull_cluster(self):
        return self._hull_cluster

    def dilated_obstacle(self, padding, id="new", name=None):
        pass

    def point_location(self, x, input_frame=Frame.GLOBAL):
        locs = [obs.point_location(x, input_frame=Frame.GLOBAL) for obs in self._obstacle_cluster] + \
               [self._hull_cluster_point_location(x)]
        if any([l < 0 for l in locs]):
            # Interior point
            return -1
        if any([l == 0 for l in locs]):
            # Boundary point
            return 0
        # Exterior point
        return 1

    def line_intersection(self, line, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        intersection_points = []
        for o in self._obstacle_cluster:
            intersection_points += o.line_intersection(line, Frame.GLOBAL, Frame.GLOBAL)
        intersection_points += self._hull_cluster_line_intersections(line)
        return intersection_points

    # TODO: Fix if needed. Currently not considering hull.
    def tangent_points(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        tp, tp_candidates = [], []
        for obs in self._obstacle_cluster:
            tp_candidates += obs.tangent_points(x, Frame.GLOBAL, Frame.GLOBAL)
        for i in range(len(tp_candidates)):
            if all([is_ccw(x, tp_candidates[i], tp_candidates[j]) for j in range(len(tp_candidates)) if j is not i]) or \
                    all([is_cw(x, tp_candidates[i], tp_candidates[j]) for j in range(len(tp_candidates)) if j is not i]):
                tp += [tp_candidates[i]]
        return tp

    def _compute_kernel(self):
        self._kernel = StarshapedPolygon(self.polygon(), xr=self.xr(), id="temp").kernel()

    def _check_convexity(self):
        self._is_convex = StarshapedPolygon(self.polygon(), xr=self.xr(), id="temp").is_convex()

    def boundary_mapping(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        # intersection_points = [p for ps in self.line_intersection([self._xr, self._xr+10*(x-self._xr)]) for p in ps]
        intersection_points = self.line_intersection([self._xr, self._xr+10*(x-self._xr)])
        if not intersection_points:
            return None
        dist_intersection_points = [np.linalg.norm(ip - self._xr) for ip in intersection_points]
        return intersection_points[np.argmax(dist_intersection_points)]

    def vel_intertial_frame(self, x):
        boundary_obs_idx = 0
        max_dist = -1
        for i, ps in enumerate(self.line_intersection([self._xr, self._xr+10*(x-self._xr)])):
            o_intersection_dist = max([np.linalg.norm(p-self._xr) for p in ps] + [-1])
            if o_intersection_dist > max_dist:
                boundary_obs_idx = i
                max_dist = o_intersection_dist
        if boundary_obs_idx >= len(self._obstacle_cluster):
            boundary_obs_idx -= len(self._obstacle_cluster)
        return self._obstacle_cluster[boundary_obs_idx].vel_intertial_frame(x)

    def normal(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL, x_is_boundary=False, type='weigthed_polygon_approx'):
        if type == 'sub_normal':
            boundary_obs_idx = 0
            max_dist = -1
            line = [self._xr, self._xr+10*(x-self._xr)]
            for i, o in enumerate(self._obstacle_cluster):
                intersection_points = o.line_intersection(line, Frame.GLOBAL, Frame.GLOBAL)
                for p in intersection_points:
                    p_dist = np.linalg.norm(p - self._xr)
                    if p_dist > max_dist:
                        max_dist = p_dist
                        boundary_obs_idx = i

            hull_ip = self._hull_cluster_line_intersections(line)
            if hull_ip:
                for p in hull_ip:
                    p_dist = np.linalg.norm(p - self._xr)
                    if p_dist > max_dist:
                        max_dist = p_dist
                        boundary_obs_idx = len(self._obstacle_cluster)
            if boundary_obs_idx < len(self._obstacle_cluster):
                return self._obstacle_cluster[boundary_obs_idx].normal(x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL)
            else:
                boundary_obs_idx -= len(self._obstacle_cluster)
                hull_vertices = np.array(self._hull_cluster.exterior.coords[:-1])
                vertex_angles = np.array([np.arctan2(v[1] - self._xr[1], v[0] - self._xr[0]) for v in hull_vertices]).flatten()
                idcs = np.argsort(vertex_angles)
                vertex_angles = vertex_angles[idcs]
                hull_vertices = hull_vertices[idcs, :]
                vertex_angles = np.hstack((vertex_angles, vertex_angles[0] + 2 * np.pi))
                hull_vertices = np.vstack((hull_vertices, hull_vertices[0, :]))

                angle = np.arctan2(x[1] - self._xr[1], x[0] - self._xr[0])
                v_idx = np.argmax(vertex_angles > angle)
                # Adjust for circular self.vertices (self.vertices[0] == self.vertices[-1])
                if v_idx == 0:
                    v_idx = -1
                n = np.array([hull_vertices[v_idx, 1] - hull_vertices[v_idx - 1, 1],
                              hull_vertices[v_idx - 1, 0] - hull_vertices[v_idx, 0]])
                n /= np.linalg.norm(n)
                return n
        elif type == 'polygon_approx' or type == 'weigthed_polygon_approx':
            if self.vertices is None:
                self._update_vertex_angles()
            angle = np.arctan2(x[1] - self._xr[1], x[0] - self._xr[0])
            v_idx = np.argmax(self.vertex_angles > angle)

            if v_idx == self.vertices.shape[0]:
                v_idx = 0

            if type == 'polygon_approx':
                n = np.array([self.vertices[v_idx, 1] - self.vertices[v_idx - 1, 1],
                              self.vertices[v_idx - 1, 0] - self.vertices[v_idx, 0]])
            else:
                edge_neighbors = [(self.vertices[(v_idx - 2 + i) % self.vertices.shape[0]],
                                   self.vertices[(v_idx - 1 + i) % self.vertices.shape[0]]) for i in range(3)]
                edge_neighbors_normal = np.array([[e[1][1] - e[0][1],
                                                   e[0][0] - e[1][0]] for e in edge_neighbors])
                edge_closest = [shapely.ops.nearest_points(shapely.geometry.LineString(e),
                                                           shapely.geometry.Point(x))[0].coords[0] for e in
                                edge_neighbors]

                dist = [np.linalg.norm(np.array(e) - x) for e in edge_closest]
                w = np.array([1 / (d + 1e-10) for d in dist])
                w /= sum(w)
                n = edge_neighbors_normal.T.dot(w)

            n /= np.linalg.norm(n)
            return n

    def set_xr(self, xr, input_frame=Frame.OBSTACLE, safe_set=False):
        super().set_xr(xr, input_frame, safe_set)
        self._update_vertex_angles()
        
    def init_plot(self, ax=None, show_reference=True, show_name=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        if "fc" not in kwargs and "facecolor" not in kwargs:
            kwargs["fc"] = 'lightgrey'

        line_handles = []

        lh, _ = draw_shapely_polygon(self.polygon(), ax=ax, **kwargs)
        line_handles += lh

        # Reference point
        line_handles += ax.plot(*self.xr(), '+', color='k') if show_reference else [None]
        # Name
        line_handles += [ax.text(*self.xr(), self._name)] if show_name else [None]
        return line_handles, ax

    def update_plot(self, line_handles):
        pass

    def draw(self, ax=None, show_reference=True, show_name=False, **kwargs):
        line_handles, ax = self.init_plot(ax, show_reference, show_name, **kwargs)
        self.update_plot(line_handles)
        return line_handles, ax

    def _hull_cluster_point_location(self, x):
        if self._hull_cluster is None:
            return 1
        x_sh = shapely.geometry.Point(x)
        if self._hull_cluster.contains(x_sh):
            return -1
        if self._hull_cluster.disjoint(x_sh):
            return 1
        return 0

    def _hull_cluster_line_intersections(self, line):
        if self._hull_cluster is None:
            return []
        line_sh = shapely.geometry.LineString(line)
        intersection_points_shapely = line_sh.intersection(self._hull_cluster.exterior)
        if intersection_points_shapely.is_empty:
            return []
        if intersection_points_shapely.geom_type == 'Point':
            return [np.array([intersection_points_shapely.x, intersection_points_shapely.y])]
        elif intersection_points_shapely.geom_type == 'MultiPoint':
            return [np.array([p.x, p.y]) for p in intersection_points_shapely.geoms]
        elif intersection_points_shapely.geom_type == 'LineString':
            return [np.array([ip[0], ip[1]]) for ip in intersection_points_shapely.coords]
        elif intersection_points_shapely.geom_type == 'MultiLineString':
            return [np.array([l[0], l[1]]) for line in intersection_points_shapely.geoms for
                                            l in line.coords]
        else:
            print("[_hull_cluster_line_intersections]: Shapely geom_type not covered!")
            print(intersection_points_shapely)

    def _update_vertex_angles(self):
        self.vertices = np.array(self._polygon.exterior.coords[:-1])
        self.circular_vertices = np.array(self._polygon.exterior.coords)
        self.vertex_angles = np.arctan2(self.vertices[:, 1] - self._xr[1], self.vertices[:, 0] - self._xr[0])
        idcs = np.argsort(self.vertex_angles)
        self.vertex_angles = self.vertex_angles[idcs]
        self.vertices = self.vertices[idcs, :]
        self.circular_vertices = np.vstack((self.vertices, self.vertices[0, :]))
        self.vertex_angles = np.hstack((self.vertex_angles, self.vertex_angles[0] + 2 * np.pi))

    def _compute_polygon_representation(self):
        obs_pol = [obs.polygon() for obs in self._obstacle_cluster]
        if self._hull_cluster is not None:
            obs_pol += [self._hull_cluster]
        self._polygon = shapely.ops.unary_union(obs_pol)
