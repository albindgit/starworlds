from obstacles import Obstacle, Frame
from utils import is_cw, is_ccw, is_collinear, tic, toc
import shapely
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Polygon(Obstacle):

    def __init__(self, polygon, **kwargs):
        # print("In Polygon obstacle")
        super().__init__(**kwargs)
        self._polygon = shapely.geometry.Polygon(polygon)
        self._polygon = shapely.ops.orient(self._polygon)
        self._pol_bounds = self._polygon.bounds
        self._compute_global_polygon_representation()
        self.vertices = np.array(self._polygon.exterior.coords[:-1])
        self.circular_vertices = np.array(self._polygon.exterior.coords)
        # self.vertices = np.array(self._polygon.exterior.coords)

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
        # cp._polygon = cp._polygon.buffer(padding, cap_style=3, join_style=3)
        cp._pol_bounds = cp._polygon.bounds
        cp.vertices = np.array(cp._polygon.exterior.coords[:-1])
        cp.circular_vertices = np.array(cp._polygon.exterior.coords)
        cp._polygon_global_pose = None
        cp._polygon_global = None
        return cp

    # def point_location(self, x, input_frame=Frame.GLOBAL):
    #     """
    #     Raycasting Algorithm to find out whether a point is in a given polygon.
    #     Performs the even-odd-rule Algorithm to find out whether a point is in a given polygon.
    #     This runs in O(n) where n is the number of edges of the polygon.
    #      *
    #     :param polygon: an array representation of the polygon where polygon[i][0] is the x Value of the i-th point and polygon[i][1] is the y Value.
    #     :param point:   an array representation of the point where point[0] is its x Value and point[1] is its y Value
    #     :return: whether the point is in the polygon (not on the edge, just turn < into <= and > into >= for that)
    #     """
    #
    #     point = self.transform(x, input_frame, Frame.OBSTACLE)
    #     xmin, ymin, xmax, ymax = self._polygon.bounds
    #     if not (xmin < point[0] < xmax and ymin < point[1] < ymax):
    #         return 1
    #     polygon = self.vertices[:-1, :]
    #
    #     # A point is in a polygon if a line from the point to infinity crosses the polygon an odd number of times
    #     odd = False
    #     # For each edge (In this case for each point of the polygon and the previous one)
    #     i = 0
    #     j = len(polygon) - 1
    #     while i < len(polygon) - 1:
    #         i = i + 1
    #         # If a line from the point into infinity crosses this edge
    #         # One point needs to be above, one below our y coordinate
    #         # ...and the edge doesn't cross our Y corrdinate before our x coordinate (but between our x coordinate and infinity)
    #
    #         if (((polygon[i, 1] > point[1]) != (polygon[j, 1] > point[1])) and (point[0] < (
    #                 (polygon[j, 0] - polygon[i, 0]) * (point[1] - polygon[i, 1]) / (polygon[j, 1] - polygon[i, 1])) +
    #                                                                             polygon[i, 0])):
    #             # Invert odd
    #             odd = not odd
    #         j = i
    #     # If the number of crossings was odd, the point is in the polygon
    #     if odd:
    #         return -1
    #     else:
    #         return 1
        # return odd

    def point_location(self, x, input_frame=Frame.GLOBAL):

        t0 = tic()
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        t1 = toc(t0) * 100
        t0 = tic()
        xmin, ymin, xmax, ymax = self._pol_bounds
        if not (xmin < x_obstacle[0] < xmax and ymin < x_obstacle[1] < ymax):
            return 1
        t2 = toc(t0) * 100
        t0 = tic()

        if False and self.is_convex():
            RIGHT = "RIGHT"
            LEFT = "LEFT"

            def get_side(a, b):
                x = cosine_sign(a, b)
                if x < 0:
                    return LEFT
                elif x > 0:
                    return RIGHT
                else:
                    return None

            def v_sub(a, b):
                return (a[0] - b[0], a[1] - b[1])

            def cosine_sign(a, b):
                return a[0] * b[1] - a[1] * b[0]


            previous_side = None
            vertices = self.vertices[:-1, :]
            vertices = self.vertices
            n_vertices = len(vertices)
            for n in range(n_vertices):
                a, b = vertices[n], vertices[(n + 1) % n_vertices]
                affine_segment = v_sub(b, a)
                affine_point = v_sub(x_obstacle, a)
                current_side = get_side(affine_segment, affine_point)
                if current_side is None:
                    return 1  # outside or over an edge
                elif previous_side is None:  # first segment
                    previous_side = current_side
                elif previous_side != current_side:
                    return 1
            # print(toc(t0))
            # t0 = tic()
            # x_sh = shapely.geometry.Point(x_obstacle)
            # self._polygon.contains(x_sh)
            return -1

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

        if False and self.is_convex():
            # vertices = self.vertices[:-1, :]
            vertices = self.vertices
            # print(vertices)
            # pl, timing = self.point_location(x_obstacle, Frame.OBSTACLE, time=1)
            # if pl < 0:
            t0 = tic()
            x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
            # ip = self.interior_point(x_obstacle, Frame.OBSTACLE)
            # if not self.exterior_point(x_obstacle, Frame.OBSTACLE):
            #     return []
            t1 = toc(t0)
            t0 = tic()
            dist_squared = [(v[0] - x_obstacle[0])**2 + (v[1] - x_obstacle[1])**2 for v in vertices]
            t2 = toc(t0)
            t0 = tic()
            closest_vert_idx = dist_squared.index(min(dist_squared))
            t3 = toc(t0)
            # t0 = tic()
            # dist_squared = np.sum(np.square(vertices - x_obstacle), axis=1)
            # t4 = toc(t0)
            # t0 = tic()
            # closest_vert_idx2 = np.argmin(dist_squared)
            # t5 = toc(t0)
            N = vertices.shape[0]

            # tp_left_idx = closest_vert_idx
            # tp_right_idx = closest_vert_idx
            # tp_left_found = False
            # tp_right_found = False
            # while not (tp_left_found and tp_right_found):
            #     if not tp_left_found:
            #         tp_left_candidate_idx = (tp_left_idx - 1) % N
            #         if is_ccw(x_obstacle, vertices[tp_left_idx, :], vertices[tp_left_candidate_idx, :]):
            #             tp_left_idx = tp_left_candidate_idx
            #         else:
            #             tp_left_found = True
            #     if not tp_right_found:
            #         tp_right_candidate_idx = (tp_right_idx + 1) % N
            #         if is_cw(x_obstacle, vertices[tp_right_idx, :], vertices[tp_right_candidate_idx, :]):
            #             tp_right_idx = tp_right_candidate_idx
            #         else:
            #             tp_right_found = True
            #
            #     # if ip and not (tp_left_idx == closest_vert_idx and tp_right_idx == closest_vert_idx):
            #     #     print("Interior point")
            #
            #
            #     # Interior point
            #     if tp_left_idx == closest_vert_idx and tp_right_idx == closest_vert_idx:
            #         # if not ip:
            #         #     _, ax = self.draw(frame=Frame.OBSTACLE)
            #         #     ax.plot(*x_obstacle, 'o')
            #         #     print(tp_left_found,tp_right_found)
            #         #     # print(self._polygon.bounds, x_obstacle)
            #         #     ax.set_xlim([min(0,x_obstacle[0]-1), max(13,x_obstacle[0]+1)])
            #         #     ax.set_ylim([min(-1,x_obstacle[1]-1), max(11,x_obstacle[1]+1)])
            #         #     ax.plot(*vertices[tp_left_idx, :], '*r')
            #         #     ax.plot(*vertices[tp_right_idx, :], '*g')
            #         #     ax.plot(*vertices[(tp_left_idx - 1) % N, :], 'dr')
            #         #     ax.plot(*vertices[(tp_right_idx + 1) % N, :], 'dg')
            #         #     for i in range(N):
            #         #         ax.text(*vertices[i, :], str(i))
            #         #     plt.show()
            #         return []
            #
            # t4 = toc(t0)


            t0 = tic()
            tp_right_idx = closest_vert_idx
            while is_cw(x_obstacle, vertices[tp_right_idx], vertices[(tp_right_idx + 1) % N]):
                tp_right_idx = (tp_right_idx + 1) % N

            tp_left_idx = closest_vert_idx
            while is_ccw(x_obstacle, vertices[tp_left_idx], vertices[(tp_left_idx - 1) % N]):
                tp_left_idx = (tp_left_idx - 1) % N
            t4 = toc(t0)

            if tp_left_idx == closest_vert_idx and tp_right_idx == closest_vert_idx:
                print(x_obstacle, vertices[tp_right_idx], vertices[(tp_right_idx+1)%N])
                print(is_cw(x_obstacle, vertices[tp_right_idx], vertices[(tp_right_idx+1)%N]))
                print(x_obstacle, vertices[tp_left_idx], vertices[(tp_left_idx-1)%N])
                print(is_ccw(x_obstacle, vertices[tp_left_idx], vertices[(tp_left_idx-1)%N]))
                return []

            t0 = tic()
            tp1 = self.transform(vertices[tp_left_idx, :], Frame.OBSTACLE, output_frame)
            tp2 = self.transform(vertices[tp_right_idx, :], Frame.OBSTACLE, output_frame)
            t5 = toc(t0)

            time_string = "{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(t1*100,t2*100,t3*100,t4*100,t5*100, (t1+t2+t3+t4+t5)*100)
            time_conv = (t1+t2+t3+t4+t5)
            # print(time_string)
            return [tp1, tp2]

            # exit()

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
        #
        tend = toc(t0)

        # print(sum([t1*100,t2*100,t3*100,t4*100,t5*100,tend*100*0]))
        # if time_conv > (t1+t2+t3+t4+t5+tend)+1:
        #     print(time_string)
        #     print("{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f},{:.2f}".format(t1*100,t2*100,t3*100,t4*100,t5*100,tend*100,(t1+t2+t3+t4+t5+tend)*100))
        #     exit()

        # _, ax = self.draw(frame=Frame.OBSTACLE)
        # ax.plot(*zip(x_obstacle,tp1_obstacle), 'r-')
        # ax.plot(*zip(x_obstacle,tp2_obstacle), 'r-')
        # ax.plot(*zip(x_obstacle,tp_planar[0]), 'g--')
        # ax.plot(*zip(x_obstacle,tp_planar[1]), 'g--')
        # # ax.plot([x_obstacle[0], tp1_obstacle[0]], [x_obstacle[1], tp1_obstacle[1]], 'r-')
        # # ax.plot([x_obstacle[0], tp2_obstacle[0]], [x_obstacle[1], tp2_obstacle[1]], 'r-')
        # # ax.plot([x_obstacle[0], tp_planar[0].x], [x_obstacle[1], tp_planar[0].y], 'g--')
        # # ax.plot([x_obstacle[0], tp_planar[1].x], [x_obstacle[1], tp_planar[1].y], 'g--')
        # ax.set_xlim([-3, 10])
        # ax.set_ylim([-3, 10])
        # plt.show()

        # if not ((np.array_equal(tp1_obstacle, vertices[tp_left_idx, :]) or np.array_equal(tp1_obstacle, vertices[tp_right_idx, :]))
        #     and (np.array_equal(tp2_obstacle, vertices[tp_left_idx, :]) or np.array_equal(tp2_obstacle, vertices[tp_right_idx, :]))):
        #     _, ax = self.draw(frame=Frame.OBSTACLE)
        #     ax.plot(*x_obstacle, 'o')
        #     ax.set_xlim([min(-2,x_obstacle[0]-1), max(2,x_obstacle[0]+1)])
        #     ax.set_ylim([min(-2,x_obstacle[1]-1), max(2,x_obstacle[1]+1)])
        #     for i in range(N):
        #         ax.text(*vertices[i, :], str(i))
        #     tp_left_candidate_idx = (tp_left_idx - 1) % N
        #     print(tp_left_idx, tp_left_candidate_idx, N)
        #     ax.plot(*vertices[tp_left_idx, :], '*r')
        #     # ax.plot(*vertices[tp_left_candidate_idx, :], 'dr')
        #     ax.plot(*vertices[tp_right_idx, :], '*g')
        #     ax.plot(*zip(x_obstacle,tp1_obstacle), 'r-')
        #     ax.plot(*zip(x_obstacle,tp2_obstacle), 'r-')
        #     plt.show()

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
