from obstacles import StarshapedObstacle, Frame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from utils import is_ccw, tic, toc
import shapely


class Ellipse(StarshapedObstacle):
    def __init__(self, a, xr=None, n_pol=20, **kwargs):
        self._a = np.array(a, float)
        self._a2 = np.square(self._a)
        self._n_pol = n_pol
        # self._area = np.pi * self._a[0] * self._a[1]
        if xr is None:
            xr = np.zeros(2)
        super().__init__(xr=xr, **kwargs)
        self.enclosing_ball_diameter = max(2*self._a[0], 2*self._a[1])

    def copy(self, id, name):
        if (id == 'duplicate' or id == 'd'):
            id = self.id()
        return Ellipse(id=id, name=name, a=self._a, xr=self._xr, n_pol=self._n_pol, motion_model=self._motion_model)

    def dilated_obstacle(self, padding, id="new", name=None):
        cp = self.copy(id, name)
        cp._a += padding
        cp._a2 = np.square(cp._a)
        cp.enclosing_ball_diameter = max(2 * cp._a[0], 2 * cp._a[1])
        if self._polygon is not None:
            cp._polygon = self._polygon.buffer(padding, cap_style=1, join_style=1)
            cp._polygon_global_pose = None
            cp._polygon_global = None
            cp._kernel = cp._polygon
        return cp

    def init_plot(self, ax=None, show_reference=True, show_name=False, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        # Default facecolor
        if "fc" not in kwargs and "facecolor" not in kwargs:
            kwargs["fc"] = 'lightgrey'
        line_handles = []
        # Boundary
        line_handles += [patches.Ellipse(xy=[0, 0], width=2*self._a[0], height=2*self._a[1], angle=0, **kwargs)]
        ax.add_patch(line_handles[-1])
        # Reference point
        line_handles += ax.plot(*self._xr, '+', color='k') if show_reference else [None]
        # Name
        line_handles += [ax.text(*self._xr, self._name)] if show_name else [None]
        return line_handles, ax

    def update_plot(self, line_handles, frame=Frame.GLOBAL):
        pos, rot = self.pos(frame), self.rot(frame)
        line_handles[0].center = pos
        line_handles[0].angle = np.rad2deg(rot)
        if line_handles[1] is not None:
            line_handles[1].set_data(*self.xr(frame))
        if line_handles[2] is not None:
            line_handles[2].set_position(self.xr(frame))

    def draw(self, frame=Frame.GLOBAL, **kwargs):
        line_handles, ax = self.init_plot(**kwargs)
        self.update_plot(line_handles, frame)
        return line_handles, ax

    def boundary_mapping(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)

        intersect_obstacle = self.line_intersection([self._xr, self._xr+1.1*self.enclosing_ball_diameter*self.reference_direction(x_obstacle, Frame.OBSTACLE, Frame.OBSTACLE)],
                                                    input_frame=Frame.OBSTACLE, output_frame=Frame.OBSTACLE)
        if not intersect_obstacle:
            return None
        if len(intersect_obstacle) == 1:
            return self.transform(intersect_obstacle[0], Frame.OBSTACLE, output_frame)
        else:
            if np.linalg.norm(intersect_obstacle[0]-x_obstacle) < np.linalg.norm(intersect_obstacle[1]-x_obstacle):
                return self.transform(intersect_obstacle[0], Frame.OBSTACLE, output_frame)
            else:
                return self.transform(intersect_obstacle[1], Frame.OBSTACLE, output_frame)

    def normal(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL, x_is_boundary=False):
        b_obstacle = self.transform(x, input_frame, Frame.OBSTACLE) if x_is_boundary else self.boundary_mapping(x, input_frame, Frame.OBSTACLE)
        if b_obstacle is None:
            return None
        n_obstacle = np.array([self._a2[1] * b_obstacle[0], self._a2[0] * b_obstacle[1]])
        n_obstacle = n_obstacle / np.linalg.norm(n_obstacle)
        return self.rotate(n_obstacle, Frame.OBSTACLE, output_frame)

    def point_location(self, x, input_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        loc = (x_obstacle[0] / self._a[0]) ** 2 + (x_obstacle[1] / self._a[1]) ** 2 - 1
        return loc

    def line_intersection(self, line, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        # Transform line points to left/right points in obstacle ellipse coordinates
        l0_obstacle = self.transform(line[0], input_frame, Frame.OBSTACLE)
        l1_obstacle = self.transform(line[1], input_frame, Frame.OBSTACLE)
        l_left_obstacle = l0_obstacle if l0_obstacle[0] < l1_obstacle[0] else l1_obstacle
        l_right_obstacle = l1_obstacle if l0_obstacle[0] < l1_obstacle[0] else l0_obstacle


        vertical_line = abs(l_right_obstacle[0]-l_left_obstacle[0]) < 1e-4
        if not vertical_line:
            # Line parameters
            m = (l_right_obstacle[1] - l_left_obstacle[1]) / (l_right_obstacle[0] - l_left_obstacle[0])
            c = l_left_obstacle[1] - m * l_left_obstacle[0]
            vertical_line = abs(m) > 100

        # Special case with vertical line
        if vertical_line:
            if l_right_obstacle[0] < -self._a[0] or l_right_obstacle[0] > self._a[0]:
                return []

            l_top_obstacle = l_right_obstacle if l_right_obstacle[1] > l_left_obstacle[1] else l_left_obstacle
            l_bottom_obstacle = l_left_obstacle if l_right_obstacle[1] > l_left_obstacle[1] else l_right_obstacle
            x_intersect_top_obstacle, x_intersect_bottom_obstacle = np.array([0, self._a[1]]), np.array([0, -self._a[1]])
            x_intersect_top = self.transform(x_intersect_top_obstacle, Frame.OBSTACLE, output_frame)
            x_intersect_bottom = self.transform(x_intersect_bottom_obstacle, Frame.OBSTACLE, output_frame)
            if l_top_obstacle[1] >= self._a[1] and l_bottom_obstacle[1] <= -self._a[1]:
                return [x_intersect_top, x_intersect_bottom]
            elif l_top_obstacle[1] >= self._a[1] and l_bottom_obstacle[1] <= self._a[1]:
                return [x_intersect_top]
            elif l_top_obstacle[1] >= -self._a[1] and l_bottom_obstacle[1] <= -self._a[1]:
                return [x_intersect_bottom]
            else:
                return []

        # obstacle ellipse coefficients at intersection with line m*x+c
        kx2 = self._a2[0] * m**2 + self._a2[1]
        kx = 2 * self._a2[0] * m * c
        k1 = self._a2[0] * (c**2 - self._a2[1])

        # TODO: Stable fix for finding intersection
        discriminant = self._a2[0] * m**2 + self._a2[1] - c**2
        if discriminant < 0:
            return []
        elif discriminant == 0:
            tmp_x = -kx / (2 * kx2)
            x_intersect_obstacle = np.array([tmp_x, m * tmp_x + c])
            return [self.transform(x_intersect_obstacle, Frame.OBSTACLE, output_frame)]
        else:
            in_sqrt = kx ** 2 / (4 * kx2 ** 2) - k1 / kx2
            if np.isclose(in_sqrt, 0):
                in_sqrt = 0
            tmp_x = -kx / (2 * kx2) - np.sqrt(in_sqrt)
            x_intersect_left_obstacle = np.array([tmp_x, m * tmp_x + c])
            tmp_x = -kx / (2 * kx2) + np.sqrt(in_sqrt)
            x_intersect_right_obstacle = np.array([tmp_x, m * tmp_x + c])
            x_intersect_left = self.transform(x_intersect_left_obstacle, Frame.OBSTACLE, output_frame)
            x_intersect_right = self.transform(x_intersect_right_obstacle, Frame.OBSTACLE, output_frame)

            if l_right_obstacle[0] < x_intersect_left_obstacle[0] or x_intersect_right_obstacle[0] < l_left_obstacle[0] or \
                    x_intersect_left_obstacle[0] < l_left_obstacle[0] < l_right_obstacle[0] < x_intersect_right_obstacle[0]:
                return []
            elif x_intersect_left_obstacle[0] < l_left_obstacle[0]:
                return [x_intersect_right]
            elif l_right_obstacle[0] < x_intersect_right_obstacle[0]:
                return [x_intersect_left]
            else:
                return [x_intersect_left, x_intersect_right]

    def tangent_points(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        if not self.exterior_point(x_obstacle, Frame.OBSTACLE):
            return []
        px, py = x_obstacle

        # Special case with vertical tangent
        a2 = self._a**2
        vertical_tangent = abs(a2[0] - px**2) < 1e-5
        if vertical_tangent:
            m2 = (py**2-a2[1]) / (2*px*py)
            x2 = (px * m2 ** 2 - py * m2) / (a2[1] / a2[0] + m2 ** 2)
            y2 = m2 * (x2 - px) + py
            tp1_obstacle = np.array([px, 0])
            tp2_obstacle = np.array([x2, y2])
        else:
            tmp = px**2 - a2[0]
            c1 = (px * py) / tmp
            c2 = (a2[1] - py**2) / tmp
            tmp = np.sqrt(c1**2+c2)
            m1, m2 = c1 + tmp, c1 - tmp
            tmp = a2[1] / a2[0]
            m1_sqr = m1**2
            m2_sqr = m2**2
            x1 = (px * m1_sqr - py * m1) / (tmp + m1_sqr)
            x2 = (px * m2_sqr - py * m2) / (tmp + m2_sqr)
            y1 = m1 * (x1 - px) + py
            y2 = m2 * (x2 - px) + py
            tp1_obstacle = np.array([x1, y1])
            tp2_obstacle = np.array([x2, y2])

        tp1 = self.transform(tp1_obstacle, Frame.OBSTACLE, output_frame)
        tp2 = self.transform(tp2_obstacle, Frame.OBSTACLE, output_frame)

        if is_ccw(x, tp1, tp2):
            tp1, tp2 = tp2, tp1

        return [tp1, tp2]

    # def area(self):
    #     return self._area

    # ------------ Private methods ------------ #
    def _check_convexity(self):
        self._is_convex = True

    def _compute_kernel(self):
        self._kernel = self._polygon

    def _compute_polygon_representation(self):
        # logprint(str(self) + ": " + str(self._polygon), 0)
        t = np.linspace(0, 2 * np.pi, self._n_pol, endpoint=False)
        a = self._a + 1e-3 # Add offset to adjust for polygon approximation
        polygon = np.vstack((a[0] * np.cos(t), a[1] * np.sin(t))).T
        self._polygon = shapely.geometry.Polygon(polygon)
