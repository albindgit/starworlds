import shapely
import numpy as np
from obstacles import Polygon
from utils import is_ccw, is_cw, line, ray, Cone, convex_hull
import matplotlib.pyplot as plt


def admissible_kernel(obstacle, x):
    # Find tangents of obstacle through x
    tps = obstacle.tangent_points(x)
    if not tps:
        # Interior point
        return None
    return Cone(x, x-tps[0], x-tps[1])


# Computes the starshaped hull of a list of obstacles for specified kernel points
def kernel_starshaped_hull(obstacles, kernel_points):
    if not type(obstacles) is list:
        if obstacles.is_convex():
            return convex_kernel_starshaped_hull(obstacles, kernel_points)
        if issubclass(obstacles.__class__, Polygon):
            return polygon_kernel_starshaped_hull(obstacles.polygon(), kernel_points)
        else:
            print("[kernel_starshaped_hull]: Bad obstacle class.")
            print(obstacles)

    sub_pols = [kernel_starshaped_hull(o, kernel_points) for o in obstacles]
    hull_polygon = shapely.ops.unary_union(sub_pols)
    if hull_polygon.is_empty:
        return None
    return hull_polygon


def convex_kernel_starshaped_hull(convex_obstacle, kernel_points):
    tps = []
    for k in kernel_points:
        tps += convex_obstacle.tangent_points(k)
    if not tps:
        return shapely.geometry.Polygon([])

    tps = np.unique(tps,axis=0)
    ch_points = np.vstack((tps, kernel_points))
    pol = convex_hull(ch_points)
    return shapely.geometry.Polygon(pol)


# TODO: Improve computational consideration
def polygon_kernel_starshaped_hull(polygon, kernel_points, debug=0):
    kernel_points = kernel_points.reshape((kernel_points.size//2, 2))

    if kernel_points.shape[0] > 2:
        # NOTE: Assumes kernel points convex
        # convex_kernel_subset = shapely.geometry.Polygon(kernel_points[ConvexHull(kernel_points).vertices, :])
        convex_kernel_subset = shapely.geometry.Polygon(kernel_points)

    # polygon_sh = polygon.polygon()  # Shapely represenation of polygon
    # vertices = np.asarray(polygon_sh.exterior.coords)[:-1, :]  # Vertices of polygon
    vertices = np.asarray(polygon.exterior.coords)[:-1, :]  # Vertices of polygon
    star_vertices = []  # Vertices of starshaped hull polygon
    v_bar = kernel_points[0].copy() # Last vertex of starshaped hull polygon
    e1_idx = 0
    e2_idx = 0
    k_centroid = np.mean(kernel_points, axis=0)
    k_included = [False] * kernel_points.shape[0]

    # Arrange vertices such that v_1 is the one with largest x-value and vendv1v2 is CCW, (assumes no collinear vertices in P)
    start_idx = np.argmax(vertices[:, 0])
    vertices = np.roll(vertices, -start_idx, axis=0)
    if is_cw(vertices[-1], vertices[0], vertices[1]):
        vertices = np.flip(vertices, axis=0)
        vertices = np.roll(vertices, 1, axis=0)
    # print("Initial sort: {:.1f}".format(toc(t0)))

    # Iterate through all vertices
    for v_idx, v in enumerate(vertices):
        adjust_e1 = False
        # Check if no ray r(v,kv) intersects with interior of polygon
        if all([ray(v, k, v).disjoint(polygon) for k in kernel_points]):
            # Add current vertex
            if kernel_points.shape[0] < 3 or not convex_kernel_subset.contains(shapely.geometry.Point(v)):
                star_vertices += [v]
            if star_vertices:
                # Intersections of lines l(k,v) and l(e1,e2)
                e1, e2 = star_vertices[e1_idx], star_vertices[e2_idx]
                e1_e2 = line(e1, e2)

                for k in kernel_points:
                    kv_e1e2_intersect = line(k,v).intersection(e1_e2)
                    # Adjust to closest intersection to e2
                    if not kv_e1e2_intersect.is_empty:
                        adjust_e1 = True
                        e1_candidate = np.array([kv_e1e2_intersect.x, kv_e1e2_intersect.y])
                        if np.linalg.norm(e2 - e1_candidate) < np.linalg.norm(e2 - star_vertices[e1_idx]):
                            star_vertices[e1_idx] = e1_candidate

            if not adjust_e1:
                for k_idx, k in enumerate(kernel_points):
                    kps = [kp for kp in kernel_points if not np.array_equal(kp, k)]
                    kv_P_intersect = line(k, v).intersection(polygon)

                    # If l(k,v) intersects interior of P
                    if not kv_P_intersect.is_empty:
                        # Find last intersection of l(k,v) and polygon boundary
                        if kv_P_intersect.geom_type == 'LineString':
                            intersection_points = [np.array([ip[0], ip[1]]) for ip in kv_P_intersect.coords]
                        elif kv_P_intersect.geom_type == 'MultiLineString':
                            intersection_points = [np.array([ip[0], ip[1]]) for l in kv_P_intersect.geoms for ip in
                                                   l.coords]
                        elif kv_P_intersect.geom_type == 'GeometryCollection':
                            intersection_points = []
                            for g in kv_P_intersect.geoms:
                                if g.geom_type == 'Point':
                                    intersection_points += [np.array(g.coords[0])]
                                if kv_P_intersect.geom_type == 'LineString':
                                    intersection_points += [np.array([ip[0], ip[1]]) for ip in g.coords]
                        else:
                            intersection_points = []

                        u = None
                        u_v = None
                        for u_candidate in intersection_points:
                            u_v = line(u_candidate, v)
                            if u_v.disjoint(polygon):
                                u = u_candidate
                                break
                        if u is None:
                            continue

                        # If no ray r(u,k'v) intersect with interior of polygon
                        if not any([ray(u, kp, v).intersects(polygon) for kp in kps]):
                            # Adjust u if l(k',v_bar) intersects l(u,v)
                            for kp in kps:
                                kpvb_uv_intersect = line(kp, v_bar).intersection(u_v)
                                if not kpvb_uv_intersect.is_empty:
                                    u = np.array([kpvb_uv_intersect.x, kpvb_uv_intersect.y])
                            # Append u to P*
                            star_vertices += [u]
                            # Update last augmented edge
                            e1_idx, e2_idx = len(star_vertices)-1, len(star_vertices)-2
                            # Swap last vertices if not CCW
                            if is_ccw(u, v, vertices[v_idx-1]):
                            # if is_ccw(v_bar, v, u):
                            # if is_cw(k_centroid, v, u):
                                star_vertices[-2], star_vertices[-1] = star_vertices[-1], star_vertices[-2]
                                e1_idx, e2_idx = e2_idx, e1_idx
                            adjust_e1 = True
                    else:
                        # Check if no ray r(k,k'v) intersect with interior of polygon
                        if (not k_included[k_idx]) and (not any([ray(k, kp, v).intersects(polygon) for kp in kps])):
                            k_included[k_idx] = True
                            # Append k to P*
                            star_vertices += [k]
                            # Update last augmented edge
                            e1_idx, e2_idx = len(star_vertices)-1, len(star_vertices)-2
                            # Swap last vertices if not CCW
                            if is_ccw(k, v, vertices[v_idx-1]):
                            # if is_cw(k_centroid, v, k):
                                star_vertices[-2], star_vertices[-1] = star_vertices[-1], star_vertices[-2]
                                e1_idx, e2_idx = e2_idx, e1_idx
                            adjust_e1 = True
            # Update v_bar
            v_bar = star_vertices[-1]

            # Visualize debug information
            if debug == 1:
                plt.plot(*k_centroid, 'ko')
                plt.plot(*polygon.exterior.xy, 'k')
                plt.plot([p[0] for p in star_vertices], [p[1] for p in star_vertices], 'g-o', linewidth=2)
                [plt.plot(*k, 'kx') for k in kernel_points]
                [plt.plot(*line(k,v).xy, 'k--') for k in kernel_points]
                if adjust_e1:
                    plt.plot(*star_vertices[e1_idx], 'ys')
                plt.show()

    # Check not added kernel points if they should be included
    for j in range(len(star_vertices)):
        v, vp = star_vertices[j - 1], star_vertices[j]
        for k_idx, k in enumerate(kernel_points):
            if (not k_included[k_idx]) and is_cw(k, v, vp):
                k_included[k_idx] = True
                # Insert k
                star_vertices = star_vertices[:j] + [k] + star_vertices[j:]
                # Visualize debug information
                if debug == 1:
                    plt.plot(*k_centroid, 'ko')
                    plt.plot(*polygon.exterior.xy, 'k')
                    plt.plot([p[0] for p in star_vertices], [p[1] for p in star_vertices], 'g-o', linewidth=2)
                    [plt.plot(*ki, 'kx') for ki in kernel_points]
                    plt.plot(*line(k, v).xy, 'r--*')
                    plt.plot(*line(k, vp).xy, 'r--*')
                    plt.plot(*k, 'go')
                    plt.show()
    # print("Final kernel check: {:.1f}".format(toc(t0)))

    if debug:
        ax = plt.gca()
        ax.plot(*polygon.exterior.xy, 'k')
        ax.plot([p[0] for p in star_vertices] + [star_vertices[0][0]],
                [p[1] for p in star_vertices] + [star_vertices[0][1]], 'g-o', linewidth=2)
        # [ax.plot(star_vertices[i][0], star_vertices[i][1], 'r*') for i in augmented_vertex_idcs]
        [ax.plot(*zip(k, sv), 'y--') for sv in star_vertices for k in kernel_points]
        ax.plot(*k_centroid, 'bs')
        plt.show()

    return shapely.geometry.Polygon(star_vertices)

