import numpy as np
import shapely.geometry
import shapely.ops
import matplotlib.pyplot as plt
from typing import List, Tuple
from utils import draw_shapely_polygon

DEFAULT_RAY_INFINITY_LENGTH = 100000.
COLLINEAR_THRESHOLD = 1e-10


def affine_transform(x, rotation, translation, inverse=False):
    if inverse:
        x_t = [x[0] - translation[0], x[1] - translation[1]]
        if rotation == 0:
            return np.array(x_t)
        c, s = np.cos(rotation), np.sin(rotation)
        return np.array([c*x_t[0] + s*x_t[1], -s*x_t[0] + c*x_t[1]])
    else:
        if rotation != 0:
            c, s = np.cos(rotation), np.sin(rotation)
            return np.array([c*x[0]-s*x[1]+translation[0], s*x[0]+c*x[1]+translation[1]])
        else:
            return np.array([x[0]+translation[0], x[1]+translation[1]])


# Line segment from a to b, excluding a and b
def line(a, b):
    return shapely.geometry.LineString([a + .0001 * (b - a), a + .9999 * (b - a)])


# Random point on the triangle with vertices a, b and c
def point_in_triangle(a, b, c):
    """
    """
    x, y = np.random.rand(2)
    q = abs(x - y)
    s, t, u = q, 0.5 * (x + y - q), 1 - 0.5 * (q + x + y)
    return s * a[0] + t * b[0] + u * c[0], s * a[1] + t * b[1] + u * c[1]

# TODO: Dynamic length of ray
# Ray emanating from a in direction of b->c
def ray(a, b, c, ray_inf_length=DEFAULT_RAY_INFINITY_LENGTH):
    return shapely.geometry.LineString([a + .0001 * (c - b), a + ray_inf_length * (c - b)])


def np_orientation_val(a, b, c):
    return (b[:, 1] - a[:, 1]) * (c[:, 0] - b[:, 0]) - (b[:, 0] - a[:, 0]) * (c[:, 1] - b[:, 1])


def orientation_val(a, b, c):
    return (b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])


def is_collinear(a, b, c):
    return abs(orientation_val(a, b, c)) < COLLINEAR_THRESHOLD


def is_cw(a, b, c):
    return orientation_val(a, b, c) > COLLINEAR_THRESHOLD


def is_ccw(a, b, c):
    return orientation_val(a, b, c) < -COLLINEAR_THRESHOLD


# TODO: Use [NOT is_cw and NOT is_ccw] instead?
# Returns true if point b is between point and b
def is_between(a, b, c):
    return np.isclose(np.linalg.norm(a-b) + np.linalg.norm(c-b), np.linalg.norm(a-c), rtol=1e-7)


def intersect(a1, a2, b1, b2):
    return (is_ccw(a1, b1, b2) != is_ccw(a2, b1, b2) and is_ccw(a1, a2, b1) != is_ccw(a1, a2, b2))


class Point:

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
        self.xy = [x, y]

    def __str__(self):
        return "Point {:s}".format(str(self.xy))

    def __iter__(self):
        return iter(self.xy)

    def __getitem__(self, item):
        return self.xy[item]

    def draw(self, ax=None, marker='o', **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        handles = ax.plot(self.x, self.y, marker=marker, **kwargs)
        return handles, ax


class Line:

    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2

    def __str__(self):
        return "Line --({:.2f},{:.2f})--({:.2f},{:.2f})--".format(self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def line_intersection(self, other: 'Line') -> Point:
        self_dx = self.p1.x - self.p2.x
        other_dx = other.p1.x - other.p2.x
        self_dy = self.p1.y - self.p2.y
        other_dy = other.p1.y - other.p2.y
        den = self_dx * other_dy - self_dy * other_dx
        if abs(den) < 1e-10:
            # Parallel or coincident lines
            return None
        tmp1 = self.p1.x * self.p2.y - self.p1.y * self.p2.x
        tmp2 = other.p1.x * other.p2.y - other.p1.y * other.p2.x
        ip_x = (tmp1 * other_dx - self_dx * tmp2) / den
        ip_y = (tmp1 * other_dy - self_dy * tmp2) / den
        return Point(ip_x, ip_y)

    def intersects(self, other):
        return (is_ccw(self.p1, other.p1, other.p1) != is_ccw(self.p2, other.p1, other.p1) and is_ccw(self.p1, self.p2, other.p1) != is_ccw(self.p1, self.p2, other.p2))


class LineSegment(Line):

    def __str__(self):
        return "Line segment ({:.2f},{:.2f})--({:.2f},{:.2f})".format(self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def line_segment_intersection(self, other: 'LineSegment') -> Point:
        self_dx = self.p1.x - self.p2.x
        other_dx = other.p1.x - other.p2.x
        self_dy = self.p1.y - self.p2.y
        other_dy = other.p1.y - other.p2.y
        p1_dx = self.p1.x - other.p1.x
        p1_dy = self.p1.y - other.p1.y
        den = self_dx * other_dy - self_dy * other_dx
        if abs(den) < 1e-10:
            # Parallel or coincident lines
            return None
        t = (p1_dx * other_dy - p1_dy * other_dx) / den
        u = (p1_dx * self_dy - p1_dy * self_dx) / den
        if t < 0 or t > 1 or u < 0 or u > 1:
            return None
        ip_x = self.p1.x - t * self_dx
        ip_y = self.p1.y - t * self_dy
        return Point(ip_x, ip_y)

class Ray(Line):

    def __str__(self):
        return "Ray ({:.2f},{:.2f})--({:.2f},{:.2f})--".format(self.p1.x, self.p1.y, self.p2.x, self.p2.y)

    def ray_intersection(self, other: 'Ray') -> Point:
        self_dx = self.p1.x - self.p2.x
        other_dx = other.p1.x - other.p2.x
        self_dy = self.p1.y - self.p2.y
        other_dy = other.p1.y - other.p2.y
        p1_dx = self.p1.x - other.p1.x
        p1_dy = self.p1.y - other.p1.y
        den = self_dx * other_dy - self_dy * other_dx
        if abs(den) < 1e-10:
            # Parallel or coincident lines
            return None
        t = (p1_dx * other_dy - p1_dy * other_dx) / den
        u = (p1_dx * self_dy - p1_dy * self_dx) / den
        if t < 0 or u < 0:
            return None
        ip_x = self.p1.x - t * self_dx
        ip_y = self.p1.y - t * self_dy
        return Point(ip_x, ip_y)

    def draw(self, ax=None, linestyle='--', color='k', markersize=16, **kwargs):
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        handles = ax.plot(*zip(self.p1, self.p2), linestyle=linestyle, color=color, **kwargs)
        orient = np.rad2deg(np.arctan2(self.p2.y-self.p1.y, self.p2.x-self.p1.x))
        handles += ax.plot(*self.p2, marker=(3, 0, orient-90), markersize=markersize, linestyle='None', color=color)
        return handles, ax


# Get intersection of line a and line b
def get_intersection(a1, a2, b1, b2):
    if not intersect(a1, a2, b1, b2):
        if is_between(b1, a1, b2):
            return a1
        if is_between(b1, a2, b2):
            return a2
        if is_between(a1, b1, a2):
            return b1
        if is_between(a1, b2, a2):
            return b2
        return None
    da = a2 - a1
    db = b2 - b1
    dp = a1 - b1
    dap = np.array([-da[1], da[0]])
    denom = np.dot(dap, db)
    num = np.dot(dap, dp)
    return (num / denom.astype(float)) * db + b1


def equilateral_triangle(centroid, side_length, rot=0):
    triangle = np.array(centroid) + np.array([[0, 1 / np.sqrt(3) * side_length],
                                               [1 / 2 * side_length, -1 / (2*np.sqrt(3)) * side_length],
                                               [-1 / 2 * side_length, -1 / (2*np.sqrt(3)) * side_length]])
    if not rot:
        return triangle
    c,s = np.cos(rot), np.sin(rot)
    return np.array([[c * triangle[0, 0] - s * triangle[0, 1], s * triangle[0, 0] + c * triangle[0, 1]],
                     [c * triangle[1, 0] - s * triangle[1, 1], s * triangle[1, 0] + c * triangle[1, 1]],
                     [c * triangle[2, 0] - s * triangle[2, 1], s * triangle[2, 0] + c * triangle[2, 1]]])


def convex_hull(points):
    n = len(points)

    # Find the leftmost point
    l = 0
    for i in range(1, n):
        if points[i][0] < points[l][0]:
            l = i
        elif points[i][0] == points[l][0]:
            if points[i][1] > points[l][1]:
                l = i

    hull = []
    '''
    Start from leftmost point, keep moving counterclockwise
    until reach the start point again. This loop runs O(h)
    times where h is number of points in result or output.
    '''
    p = l
    while (True):
        # Add current point to result
        hull.append([points[p][0], points[p][1]])

        '''
        Search for a point 'q' such that orientation(p, q,
        x) is counterclockwise for all points 'x'. The idea
        is to keep track of last visited most counterclock-
        wise point in q. If any point 'i' is more counterclock-
        wise than q, then update q.
        '''
        q = (p + 1) % n

        for i in range(n):

            # If i is more counterclockwise
            # than current q, then update q
            if is_ccw(points[p], points[i], points[q]):
                q = i
        '''
        Now q is the most counterclockwise with respect to p
        Set p as q for next iteration, so that q is added to
        result 'hull'
        '''
        p = q

        # While we don't come to first point
        if (p == l):
            break

    return hull


class Cone:
    bb_width = 1e6
    bottom_right = Point(bb_width, -bb_width)
    top_right = Point(bb_width, bb_width)
    bottom_left = Point(-bb_width, -bb_width)
    top_left = Point(-bb_width, bb_width)
    right_line = Line(bottom_right, top_right)
    top_line = Line(top_left, top_right)
    left_line = Line(top_left, bottom_left)
    bottom_line = Line(bottom_left, bottom_right)
    bb_edges = [bottom_line, right_line, top_line, left_line]
    bb_corners = [bottom_right.xy, top_right.xy, top_left.xy, bottom_left.xy]
    bb_corner_angles = np.pi / 4 * np.array([1, 3, 5, 7])

    def __init__(self, apex, dir1, dir2):
        self.apex = np.array(apex)
        self.dir1 = np.array(dir1)
        self.dir2 = np.array(dir2)
        self.is_convex = is_ccw([0, 0], self.dir1, self.dir2)
        self.ray1 = Ray(Point(*self.apex), Point(*(self.apex+self.dir1)))
        self.ray2 = Ray(Point(*self.apex), Point(*(self.apex+self.dir2)))

    def __str__(self):
        return "Cone: ({:s}, {:s}, {:s})".format(str(self.apex), str(self.dir1), str(self.dir2))

    def polygon(self) -> shapely.geometry.Polygon:

        angle1 = np.arctan2(self.dir1[1], self.dir1[0]) + np.pi
        angle2 = np.arctan2(self.dir2[1], self.dir2[0]) + np.pi

        if Cone.bb_corner_angles[0] <= angle1 < Cone.bb_corner_angles[1]:
            ray1_intersection_idx = 0
        elif  Cone.bb_corner_angles[1] <= angle1 < Cone.bb_corner_angles[2]:
            ray1_intersection_idx = 1
        elif  Cone.bb_corner_angles[2] <= angle1 < Cone.bb_corner_angles[3]:
            ray1_intersection_idx = 2
        else:
            ray1_intersection_idx = 3

        if Cone.bb_corner_angles[0] <= angle2 < Cone.bb_corner_angles[1]:
            ray2_intersection_idx = 0
        elif  Cone.bb_corner_angles[1] <= angle2 < Cone.bb_corner_angles[2]:
            ray2_intersection_idx = 1
        elif  Cone.bb_corner_angles[2] <= angle2 < Cone.bb_corner_angles[3]:
            ray2_intersection_idx = 2
        else:
            ray2_intersection_idx = 3

        r1_border = self.ray1.ray_intersection(Cone.bb_edges[ray1_intersection_idx])
        r2_border = self.ray2.ray_intersection(Cone.bb_edges[ray2_intersection_idx])
        if r1_border is None:
            ray1_intersection_idx = (ray1_intersection_idx + 1) % 4
            r1_border = self.ray1.ray_intersection(Cone.bb_edges[ray1_intersection_idx])
            if r1_border is None:
                ray1_intersection_idx = (ray1_intersection_idx + 2) % 4
                r1_border = self.ray1.ray_intersection(Cone.bb_edges[ray1_intersection_idx])
                if r1_border is None:
                    print("SOMETHING WRONG!")
        if r2_border is None:
            ray2_intersection_idx = (ray2_intersection_idx + 1) % 4
            r2_border = self.ray2.ray_intersection(Cone.bb_edges[ray2_intersection_idx])
            if r2_border is None:
                ray2_intersection_idx = (ray2_intersection_idx + 2) % 4
                r2_border = self.ray2.ray_intersection(Cone.bb_edges[ray2_intersection_idx])
                if r2_border is None:
                    print("SOMETHING WRONG!")

        vertices = [self.apex, r1_border.xy]
        c_idx = ray1_intersection_idx
        if not (self.is_convex and (ray1_intersection_idx == ray2_intersection_idx)):
            while True:
                vertices += [Cone.bb_corners[c_idx]]
                c_idx = (c_idx + 1) % 4
                if c_idx == ray2_intersection_idx:
                    break
        vertices += [r2_border.xy]

        return shapely.geometry.Polygon(vertices)

    def point_in_cone(self, x):
        if self.is_convex:
            return is_ccw(self.apex, self.ray1.p2, x) and is_cw(self.apex, self.ray2.p2, x)
        else:
            return not (is_ccw(self.apex, self.ray2.p2, x) and is_cw(self.apex, self.ray1.p2, x))

    def intersection(self, other: 'Cone', same_apex: bool=False) -> shapely.geometry.Polygon:
        if same_apex:
            intersection_list = self.intersection_same_apex(other)
            if len(intersection_list) == 1:
                return intersection_list[0].polygon()
            else:
                return shapely.ops.unary_union([i.polygon() for i in intersection_list])
        else:
            return self.polygon().intersection(other.polygon())

    def intersection_same_apex(self, other: 'Cone') -> List['Cone']:
        other_dir1_in_self = self.point_in_cone(other.apex+other.dir1)
        other_dir2_in_self = self.point_in_cone(other.apex+other.dir2)
        self_dir1_in_other = other.point_in_cone(self.apex+self.dir1)

        if other_dir1_in_self and other_dir2_in_self:
            # Check several cones
            if self_dir1_in_other:
                return [Cone(self.apex, self.dir1, other.dir2),
                        Cone(self.apex, other.dir1, self.dir2)]
            else:
                return [other]
        elif other_dir1_in_self:
            return [Cone(self.apex, other.dir1, self.dir2)]
        elif other_dir2_in_self:
            return [Cone(self.apex, self.dir1, other.dir2)]
        elif self_dir1_in_other:
            return [self]
        else:
            return []

    @staticmethod
    def list_intersection(cones: List['Cone'], same_apex=False) -> shapely.geometry.Polygon:
        if not same_apex:
            intersection = cones[0].polygon()
            for c in cones[1:]:
                intersection = intersection.intersection(c.polygon())
            return intersection

        # List of cones
        cones_intersect = [cones[0]]
        for c in cones[1:]:
            # Find intersection of current cones and next in list
            cones_intersect_new = []
            for i, ci in enumerate(cones_intersect):
                cones_intersect_new += ci.intersection_same_apex(c)
            cones_intersect = cones_intersect_new

            # If empty intersection
            if not cones_intersect:
                return shapely.geometry.Polygon([])

        if len(cones_intersect) == 1:
            ret = cones_intersect[0].polygon()
            return ret
        else:
            return shapely.ops.unary_union([c.polygon() for c in cones_intersect])

    def draw(self, ax=None, ray_color='k', **kwargs):
        handles, ax = draw_shapely_polygon(self.polygon(), ax=ax, **kwargs)
        if ray_color is not None:
            handles += ax.plot(*zip(self.apex, self.apex + Cone.bb_width * self.dir1), linestyle='--', color=ray_color)
            handles += ax.plot(*zip(self.apex, self.apex + Cone.bb_width * self.dir2), linestyle='-', color=ray_color)
        return handles, ax


def generate_convex_polygon(n, box=None):
    if box is None:
        box = [1, 1]

    # Generate two lists of random X and Y coordinates
    xPool = [box[0]*np.random.uniform() for i in range(n)]
    yPool = [box[0]*np.random.uniform() for i in range(n)]

    # Sort them
    xPool = np.sort(xPool)
    yPool = np.sort(yPool)

    # Isolate the extreme points
    minX = xPool[0]
    maxX = xPool[-1]
    minY = yPool[0]
    maxY = yPool[-1]

    # Divide the interior points into two chains & Extract the vector components
    lastTop = minX
    lastBot = minX

    xVec, yVec = [], []
    for i in range(1, n-1):
        if np.random.randint(2):
            xVec += [xPool[i] - lastTop]
            lastTop = xPool[i]
        else:
            xVec += [lastBot - xPool[i]]
            lastBot = xPool[i]

    xVec += [maxX - lastTop]
    xVec += [lastBot - maxX]

    lastLeft = minY
    lastRight = minY

    for i in range(1, n - 1):
        if np.random.randint(2):
            yVec += [yPool[i] - lastLeft]
            lastLeft = yPool[i]
        else:
            yVec += [lastRight - yPool[i]]
            lastRight = yPool[i]

    yVec += [maxY - lastLeft]
    yVec += [lastRight - maxY]

    # Randomly pair up the X- and Y-components
    np.random.shuffle(yVec)


    # Combine the paired up components into vectors
    # vec = [[xVec[i], yVec[i]] for i in range(n)]

    # Sort the vectors by angle
    angle = [np.arctan2(yVec[i], xVec[i]) for i in range(n)]
    xVec = [x for _, x in sorted(zip(angle, xVec))]
    yVec = [y for _, y in sorted(zip(angle, yVec))]

    # Lay them end-to-end
    x, y = 0, 0
    minPolygonX, minPolygonY = 0, 0
    points = []

    for i in range(n):
        points += [[x, y]]
        x += xVec[i]
        y += yVec[i]

        minPolygonX = min(minPolygonX, x)
        minPolygonY = min(minPolygonY, y)

    xShift = minX - minPolygonX
    yShift = minY - minPolygonY


    # Move the polygon to have center in origin
    for i in range(n):
        points[i][0] += xShift - box[0]/2
        points[i][1] += yShift - box[1]/2

    xr = np.mean(points, axis=0)
    for i in range(n):
        points[i][0] -= xr[0]
        points[i][1] -= xr[1]

    if not shapely.geometry.box(-box[0],-box[1],box[0],box[1],).contains(shapely.geometry.Polygon(points)):
        _, ax = plt.subplots()
        draw_shapely_polygon(shapely.geometry.Polygon(points), ax=ax)
        ax.set_xlim([-2*box[0], 2*box[0]])
        ax.set_ylim([-2*box[1], 2*box[1]])
        plt.show()

    return points

def generate_star_polygon(center: Tuple[float, float], avg_radius: float,
                     irregularity: float, spikiness: float,
                     num_vertices: int) -> List[Tuple[float, float]]:
    """
    Start with the center of the polygon at center, then creates the
    polygon by sampling points on a circle around the center.
    Random noise is added by varying the angular spacing between
    sequential points, and by varying the radial distance of each
    point from the centre.

    Args:
        center (Tuple[float, float]):
            a pair representing the center of the circumference used
            to generate the polygon.
        avg_radius (float):
            the average radius (distance of each generated vertex to
            the center of the circumference) used to generate points
            with a normal distribution.
        irregularity (float):
            variance of the spacing of the angles between consecutive
            vertices.
        spikiness (float):
            variance of the distance of each vertex to the center of
            the circumference.
        num_vertices (int):
            the number of vertices of the polygon.
    Returns:
        List[Tuple[float, float]]: list of vertices, in CCW order.
    """
    # Parameter check
    if irregularity < 0 or irregularity > 1:
        raise ValueError("Irregularity must be between 0 and 1.")
    if spikiness < 0 or spikiness > 1:
        raise ValueError("Spikiness must be between 0 and 1.")

    irregularity *= 2 * np.pi / num_vertices
    spikiness *= avg_radius
    angle_steps = random_angle_steps(num_vertices, irregularity)

    # now generate the points
    points = []
    angle = np.random.uniform(0, 2 * np.pi)
    for i in range(num_vertices):
        radius = np.clip(np.random.normal(avg_radius, spikiness), 0, 2 * avg_radius)
        point = (center[0] + radius * np.cos(angle),
                 center[1] + radius * np.sin(angle))
        points.append(point)
        angle += angle_steps[i]

    return points

def random_angle_steps(steps: int, irregularity: float) -> List[float]:
    """Generates the division of a circumference in random angles.

    Args:
        steps (int):
            the number of angles to generate.
        irregularity (float):
            variance of the spacing of the angles between consecutive vertices.
    Returns:
        List[float]: the list of the random angles.
    """
    # generate n angle steps
    angles = []
    lower = (2 * np.pi / steps) - irregularity
    upper = (2 * np.pi / steps) + irregularity
    cumsum = 0
    for i in range(steps):
        angle = np.random.uniform(lower, upper)
        angles.append(angle)
        cumsum += angle

    # normalize the steps so that point 0 and point n+1 are the same
    cumsum /= (2 * np.pi)
    for i in range(steps):
        angles[i] /= cumsum
    return angles
