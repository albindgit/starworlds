import shapely
import numpy as np
from obstacles import Frame, StarshapedPrimitiveCombination, Ellipse, StarshapedPolygon
from utils import is_ccw, is_collinear, equilateral_triangle, Cone, tic, toc, draw_shapely_polygon
from scipy.spatial import ConvexHull
import starshaped_hull as sh
import matplotlib.pyplot as plt


class ObstacleCluster:
    def __init__(self, obstacles):
        self.name = '_'.join([str(o.id()) for o in obstacles])
        self.obstacles = obstacles
        self.cluster_obstacle = None if len(obstacles) > 1 else obstacles[0]
        self.kernel_points = None
        self.admissible_kernel = None
        self._polygon = None
        self._polygon_excluding_hull = None

    def __str__(self):
        return self.name

    def polygon(self):
        if self._polygon is None:
            if self.cluster_obstacle is None:
                print("[OBSTACLE CLUSTER]: WARNING, cluster_obstacle must be defined before accessing polygon.")
            else:
                self._polygon = self.cluster_obstacle.polygon()
        return self._polygon

    def polygon_excluding_hull(self):
        if self._polygon_excluding_hull is None:
            self._polygon_excluding_hull = shapely.ops.unary_union([o.polygon() for o in self.obstacles])
        return self._polygon_excluding_hull

    def draw(self, ax=None):
        if self.cluster_obstacle is not None:
            ax, _ = self.cluster_obstacle.draw(ax=ax, fc="green")
        for obs in self.obstacles:
            ax, _ = obs.draw(ax=ax)
        return ax, _


def get_intersection_clusters(clusters):
    No = len(clusters)
    intersection_idcs = []

    # Use polygon approximations for intersection check
    cluster_polygons = [cl.polygon() for cl in clusters]

    # Find intersections
    intersections_exist = False
    for i in range(No):
        intersection_idcs += [[i]]
        for j in range(i + 1, No):
            if cluster_polygons[i].intersects(cluster_polygons[j]):
                intersection_idcs[i] += [j]
                intersections_exist = True

    if not intersections_exist:
        return clusters, intersections_exist

    # Cluster intersecting obstacles
    for i in range(No - 1, 0, -1):
        for j in range(i - 1, -1, -1):
            found = False
            for l_j in intersection_idcs[j]:
                if l_j in intersection_idcs[i]:
                    found = True
                    break
            if found:
                intersection_idcs[j] = list(set(intersection_idcs[j] + intersection_idcs[i]))
                intersection_idcs[i] = []
                break

    # Create obstacle clusters
    cluster_obstacles = [cl.obstacles for cl in clusters]
    new_clusters = []
    for i in range(No):
        if intersection_idcs[i]:
            new_clusters += [ObstacleCluster([o for j in intersection_idcs[i] for o in cluster_obstacles[j]])]

    return new_clusters, intersections_exist


def compute_kernel_points(cl, x, xg, epsilon, cl_prev, workspace):
    triangle_center_prev = np.mean(cl_prev.kernel_points, axis=0) if cl_prev else None

    t0 = tic()
    ts = {}
    # Find triangle center selection set (TCSS)
    tcss = cl.admissible_kernel
    # If obstacle is in exterior of workspace limit TCSS to workspace exterior
    if not cl.polygon_excluding_hull().within(workspace.polygon()):
        tcss_tmp = tcss.difference(workspace.polygon())
        tcss_tmp = tcss_tmp.intersection(cl.polygon_excluding_hull()) # NOTE: Added for exlcluding undesired extremas in other workspace exterior than close to obstacles
        if tcss_tmp.area > 1e-6:
            tcss = tcss_tmp
    tcss_tmp = tcss
    ts['ws check'] = toc(t0)
    # Try to use intersection of all obstacle kernels in cluster if all starshaped
    if all([o.is_starshaped() for o in cl.obstacles]):
        for o in cl.obstacles:
            tcss_tmp = tcss_tmp.intersection(o.kernel())
    ts['kernel intersection'] = toc(t0)-list(ts.values())[-1]
    # Else, try to use union of all obstacles in cluster
    if tcss_tmp.area < 1e-6:
        tcss_tmp = tcss.intersection(cl.polygon_excluding_hull())
    ts['cluster intersection'] = toc(t0)-list(ts.values())[-1]
    if tcss_tmp.area > 1e-6:
        tcss = tcss_tmp

    # If not tc from previous iteraion, use closest point to TCSS in ad ker as triangle center
    if triangle_center_prev is None:
        tc, _ = shapely.ops.nearest_points(cl.admissible_kernel, tcss.centroid)
        triangle_center = np.array(tc.coords[0])
        while is_collinear(x, xg, triangle_center):
            triangle_center += np.random.uniform(-1e-4, 1e-4, 2)
    # Else, use previous triangle center if still in selection set
    elif tcss.contains(shapely.geometry.Point(triangle_center_prev))\
            and not is_collinear(x, xg, triangle_center_prev):
        triangle_center = triangle_center_prev
    # Else, try to maintain triangle center on same side of l(x,xg) as previous time step
    else:
        x_xg_line = shapely.geometry.LineString([x, xg])
        splitted_tcss = shapely.ops.split(tcss, x_xg_line).geoms
        triangle_center_selection_set = splitted_tcss[0]
        if len(splitted_tcss) > 1:
            for i in range(1, len(splitted_tcss)):
                if is_ccw(x, xg, splitted_tcss[i].centroid.coords[0]) == is_ccw(x, xg, triangle_center_prev):
                    triangle_center_selection_set = splitted_tcss[i]
                    break
        tc, _ = shapely.ops.nearest_points(cl.admissible_kernel, triangle_center_selection_set.centroid)
        triangle_center = np.array(tc.coords[0])

    ts['tc selection'] = toc(t0)-list(ts.values())[-1]

    # if cl.name == '5_6_7':
    #     hs, _ = draw_shapely_polygon(tcss, plt.gca(), fc='g')
    #     hs += plt.plot(*triangle_center, 'kd')
    #     while not plt.waitforbuttonpress(): pass
    #     [h.remove() for h in hs]

    # Select kernel points as largest equilateral triangle in TCSS (with maximum side length epsilon)
    if tcss.geom_type == 'Polygon':
        dist = tcss.exterior.distance(shapely.geometry.Point(triangle_center))
    else:
        tc = shapely.geometry.Point(triangle_center)
        dist = min([p.exterior.distance(tc) for p in tcss.geoms])
    triangle_length = min(epsilon, 0.9 * dist)
    kernel_points = equilateral_triangle(triangle_center, triangle_length)
    ts['triangle generation'] = toc(t0)-list(ts.values())[-1]
    tot_time = sum(ts.values())
    for k in ts.keys():
        ts[k] = int(ts[k] / tot_time * 100)
    # print(ts)
    return kernel_points

def extract_cluster(cl, cl_list):
    if cl_list is None:
        return None
    for cl_i in cl_list:
        if cl.name == cl_i.name:
            return cl_i
    return None

# Input: Convex obstacles, excluding points x and xg, kernel width epsilon
def cluster_and_starify(obstacles, x, xg, epsilon, workspace=None, max_compute_time=np.inf, previous_clusters=None,
                        make_convex=False, exclude_obstacles=False, max_iterations=np.inf, verbose=False,
                        timing_verbose=False, return_history=False):
    t0 = tic()

    if workspace is None:
        workspace = Ellipse([1e10, 1e10])

    # Exit flags
    INCOMPLETE = 0
    COMPLETE = 1
    MAX_COMPUTE_TIME_CONVEX_HULL = 2

    # Variable initialization
    kernel_time, hull_time, cluster_time, convex_time = [0.], [0.], [0.], 0.
    adm_ker_robot_cones = {obs.id(): None for obs in obstacles}
    adm_ker_goal_cones = {obs.id(): None for obs in obstacles}
    adm_ker_obstacles = {obs.id(): {} for obs in obstacles}
    cluster_iterations = []
    n_iter = 0

    def default_result():
        if return_history:
            cl_history = cluster_iterations if n_iter > 0 else [ObstacleCluster([o]) for o in obstacles]
            return [ObstacleCluster([o]) for o in obstacles], [0.] * 4, INCOMPLETE, n_iter, cl_history
        else:
            return [ObstacleCluster([o]) for o in obstacles], [0.] * 4, INCOMPLETE, n_iter

    # Compute admissible kernel for all obstacles
    for o in obstacles:
        if max_compute_time < toc(t0):
            return default_result()
        # Find admissible kernel
        adm_ker_robot_cones[o.id()] = sh.admissible_kernel(o, x)
        adm_ker_goal_cones[o.id()] = sh.admissible_kernel(o, xg)

        if adm_ker_robot_cones[o.id()] is None:
            if verbose:
                print("[Cluster and Starify]: Robot position is not a free exterior point of obstacle " + str(o.id()))
            return default_result()

        if adm_ker_goal_cones[o.id()] is None:
            if verbose:
                print("[Cluster and Starify]: Goal position is not a free exterior point of obstacle " + str(o.id()))
            return default_result()

        # Admissible kernel when excluding points of other obstacles
        if exclude_obstacles:
            for o_ex in obstacles:
                if o_ex.id() == o.id():
                    continue
                o_x_exclude = o_ex.extreme_points()
                if not all([o.exterior_point(x_ex) for x_ex in o_x_exclude]):
                    adm_ker_obstacles[o.id()][o_ex.id()] = None
                    continue
                adm_ker_obstacles[o.id()][o_ex.id()] = sh.admissible_kernel(o, o_x_exclude[0]).polygon()
                for v in o_x_exclude[1:]:
                    adm_ker_obstacles[o.id()][o_ex.id()] = adm_ker_obstacles[o.id()][o_ex.id()].intersection(
                        sh.admissible_kernel(o, v).polygon())
    init_kernel_time = toc(t0)

    # -- First iteration -- #

    ker_sel_time, hull_compute_time, star_obj_time = [0], [0], [0]
    # Initialize clusters as single obstacles
    clusters = []
    for o in obstacles:
        # Ensure not xr in l(x,xg)
        # while is_collinear(x, o.xr(), xg):
        #     o.set_xr(o.xr(output_frame=Frame.OBSTACLE) + np.random.normal(0, 0.01, 2))
        cl = ObstacleCluster([o])

        # New---
        t1 = tic()
        cl.admissible_kernel = adm_ker_robot_cones[o.id()].intersection(adm_ker_goal_cones[o.id()])
        kernel_time[0] += toc(t1)

        t1 = tic()
        cl_prev = extract_cluster(cl, previous_clusters)
        # if cl_prev is None:
        #     cl_prev.kernel_points = equilateral_triangle(o.xr(), epsilon)
        cl.kernel_points = compute_kernel_points(cl, x, xg, epsilon, cl_prev, workspace=workspace)
        ker_sel_time[0] += toc(t1)
        t1 = tic()

        # -- Compute starshaped hull of cluster
        cl_id = "new" if cl_prev is None else cl_prev.cluster_obstacle.id()
        cluster_hull_extensions = sh.kernel_starshaped_hull(cl.obstacles, cl.kernel_points)
        hull_compute_time[0] += toc(t1)
        t1 = tic()
        k_centroid = np.mean(cl.kernel_points, axis=0)
        if cluster_hull_extensions is None:
            cl.cluster_obstacle = o
            cl.cluster_obstacle.set_xr(k_centroid, input_frame=Frame.GLOBAL)
        else:
        # Non-starshaped polygons are included in the cluster hull
        # cl_obstacles = [o] if o.is_starshaped() else []
            cl.cluster_obstacle = StarshapedPrimitiveCombination(cl.obstacles, cluster_hull_extensions, xr=k_centroid,
                                                             id=cl_id)
        star_obj_time[0] += toc(t1)
        # cl.polygon()
        # if o.is_starshaped and o.kernel().contains(shapely.geometry.Point(k_centroid)):
        #     cl.cluster_obstacle
        # cl.cluster_obstacle = StarshapedPolygon(cl._polygon, xr=k_centroid, id=o.id())
        # hull_time[0] += toc(t1)


        clusters += [cl]


        # if not o.is_starshaped():
        #     cl = clusters[-1]
        #     t1 = tic()
        #     cl.admissible_kernel = adm_ker_robot_cones[o.id()].intersection(adm_ker_goal_cones[o.id()])
        #     kernel_time[0] += toc(t1)
        #
        #     t1 = tic()
        #     cl_prev = None
        #     if previous_clusters:
        #         for p_cl in previous_clusters:
        #             if cl.name == p_cl.name:
        #                 cl_prev = p_cl
        #     cl.kernel_points = compute_kernel_points(cl, x, xg, epsilon, cl_prev, workspace=workspace)
        #     # -- Compute starshaped hull of cluster
        #     k_centroid = np.mean(cl.kernel_points, axis=0)
        #     cl._polygon = sh.kernel_starshaped_hull(o, cl.kernel_points)
        #     cl.cluster_obstacle = StarshapedPolygon(cl._polygon, xr=k_centroid, id=o.id())
        #     hull_time[0] += toc(t1)
    hull_time[0] = ker_sel_time[0] + hull_compute_time[0] + star_obj_time[0]

    # Set cluster history
    cluster_history = {cl.name: cl for cl in clusters}
    if return_history:
        cluster_iterations += [clusters]

    # -- Cluster obstacles such that no intersection exists
    t1 = tic()
    clusters, intersection_exists = get_intersection_clusters(clusters)
    cluster_time[0] = toc(t1)

    n_iter = 1
    # -- End first iteration -- #

    while intersection_exists:
        kernel_time += [0.]
        hull_time += [0.]
        cluster_time += [0.]
        ker_sel_time += [0.]
        hull_compute_time += [0.]
        star_obj_time += [0.]

        # Check compute time
        if max_compute_time < toc(t0):
            if verbose:
                print("[Cluster and Starify]: Max compute time.")
            cluster_iterations += [clusters]
            return default_result()

        # Find starshaped obstacle representation for each cluster
        for i, cl in enumerate(clusters):
            # If cluster already calculated keep it
            if cl.name in cluster_history:
                clusters[i] = cluster_history[cl.name]
                continue

            # ----------- Admissible Kernel ----------- #
            t1 = tic()

            # If cluster is two convex obstacles
            # if len(cl.obstacles) == 2 and cl.obstacles[0].is_convex() and cl.obstacles[1].is_convex():
            #     cl.admissible_kernel = cl.obstacles[0].polygon().intersection(cl.obstacles[1].polygon()).buffer(0.01)
            # else:
            adm_ker_robot = Cone.list_intersection([adm_ker_robot_cones[o.id()] for o in cl.obstacles], same_apex=True)
            adm_ker_goal = Cone.list_intersection([adm_ker_goal_cones[o.id()] for o in cl.obstacles], same_apex=True)
            cl.admissible_kernel = adm_ker_robot.intersection(adm_ker_goal)

            if cl.admissible_kernel.is_empty or cl.admissible_kernel.area < 1e-6:
                if verbose:
                    print("[Cluster and Starify]: Could not find disjoint starshaped obstacles. Admissible kernel empty for the cluster " + cl.name)
                cluster_iterations += [clusters]
                return default_result()

            # Exclude other obstacles
            adm_ker_o_ex = cl.admissible_kernel
            if exclude_obstacles:
                for o in cl.obstacles:
                    for o_ex in obstacles:
                        if o_ex.id() == o.id() or adm_ker_obstacles[o.id()][o_ex.id()] is None:
                            continue
                        adm_ker_o_ex = adm_ker_o_ex.intersection(adm_ker_obstacles[o.id()][o_ex.id()])
                if not (adm_ker_o_ex.is_empty or adm_ker_o_ex.area < 1e-6):
                    cl.admissible_kernel = adm_ker_o_ex

            kernel_time[n_iter] += toc(t1)
            # ----------- End Admissible Kernel ----------- #

            # ----------- Starshaped Hull ----------- #
            t1 = tic()
            # Check if cluster exist in previous cluster
            cl_prev = None
            if previous_clusters:
                for p_cl in previous_clusters:
                    if cl.name == p_cl.name:
                        cl_prev = p_cl

            # -- Kernel points selection
            cl.kernel_points = compute_kernel_points(cl, x, xg, epsilon, cl_prev, workspace=workspace)
            ker_sel_time[n_iter] += toc(t1)
            t1 = tic()

            # -- Compute starshaped hull of cluster
            cl_id = "new" if cl_prev is None else cl_prev.cluster_obstacle.id()
            cluster_hull_extensions = sh.kernel_starshaped_hull(cl.obstacles, cl.kernel_points)
            hull_compute_time[n_iter] += toc(t1)

            t1 = tic()
            k_centroid = np.mean(cl.kernel_points, axis=0)
            # Non-starshaped polygons are included in the cluster hull
            cl_obstacles = [o for o in cl.obstacles if o.is_starshaped()]
            cl.cluster_obstacle = StarshapedPrimitiveCombination(cl_obstacles, cluster_hull_extensions, xr=k_centroid, id=cl_id)
            cl.polygon()

            star_obj_time[n_iter] += toc(t1)
            # ----------- End Starshaped Hull ----------- #

            # -- Add cluster to history
            cluster_history[cl.name] = cl

        hull_time[n_iter] = ker_sel_time[n_iter] + hull_compute_time[n_iter] + star_obj_time[n_iter]

        if return_history:
            cluster_iterations += [clusters]

        # ----------- Clustering ----------- #
        t1 = tic()
        # -- Cluster star obstacles such that no intersection exists
        clusters, intersection_exists = get_intersection_clusters(clusters)
        cluster_time[n_iter] = toc(t1)
        # ----------- End Clustering ----------- #

        n_iter += 1

        if n_iter >= max_iterations:
            break

    # ----------- Make Convex ----------- #

    if make_convex:
        # Make convex if no intersection occurs
        t1 = tic()
        for j, cl in enumerate(clusters):
            if not cl.cluster_obstacle.is_convex():
                # Check compute time
                if max_compute_time < toc(t0):
                    if verbose:
                        print("[Cluster and Starify]: Max compute time in convex hull.")
                    return clusters, [sum(cluster_time), init_kernel_time+sum(kernel_time), sum(hull_time), toc(t1)], MAX_COMPUTE_TIME_CONVEX_HULL, n_iter

                v = np.array(cl.polygon().exterior.coords)[:-1, :]
                hull = ConvexHull(v)
                hull_polygon = shapely.geometry.Polygon(v[hull.vertices, :])
                if not any([hull_polygon.contains(shapely.geometry.Point(x_ex)) for x_ex in [x, xg]]) and not any(
                        [hull_polygon.intersects(clusters[k].polygon()) for k in range(len(clusters)) if k != j]):
                    # clusters[j].cluster_obstacle = StarshapedPrimitiveCombination(cl.obstacles, hull_polygon, cl.cluster_obstacle.xr(Frame.GLOBAL))
                    clusters[j].cluster_obstacle = StarshapedPolygon(hull_polygon, xr=cl.cluster_obstacle.xr(Frame.GLOBAL))

        convex_time = toc(t1)
    # ----------- End Make Convex ----------- #

    timing_vec = [sum(cluster_time), init_kernel_time+sum(kernel_time), sum(hull_time), convex_time]
    if timing_verbose:

        print("------------\nTotal timing (Cluster/AdmKer/StHull/ConvHull): {:.1f} [{:.1f}, {:.1f}, {:.1f}, {:.1f}]".format(sum(timing_vec), *timing_vec))
        print("Init kernel timing: {:.1f}".format(init_kernel_time))
        for i in range(n_iter):
            print("Iteration {} timing (Cluster/AdmKer/StHull): [{:.1f}, {:.1f}, {:.1f}]".format(i, cluster_time[i], kernel_time[i], hull_time[i]))
            print("\t Hull timing divide: (KerSel/Hull/Obj) calculation [{:.1f}, {:.1f}, {:.1f}]".format(ker_sel_time[i], hull_compute_time[i], star_obj_time[i]))

    if return_history:
        return clusters, timing_vec, COMPLETE, n_iter, cluster_iterations
    else:
        return clusters, timing_vec, COMPLETE, n_iter


def draw_clustering(clusters, p, pg, xlim=None, ylim=None):
    color = plt.cm.rainbow(np.linspace(0, 1, len(clusters)))
    fig, ax = plt.subplots()
    for i, cl in enumerate(clusters):
        [o.draw(ax=ax, fc=color[i], show_reference=False, ec='k', alpha=0.8) for o in cl.obstacles]
    ax.plot(*p, 'ko')
    ax.plot(*pg, 'k*')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def draw_star_hull(clusters, p, pg, xlim=None, ylim=None):
    fig, ax = plt.subplots()
    for cl in clusters:
        if cl.cluster_obstacle is not None:
            cl.cluster_obstacle.draw(ax=ax, fc='g', alpha=0.8)
            draw_shapely_polygon(cl.polygon_excluding_hull(), ax=ax, fc='lightgrey', ec='k')
        else:
            draw_shapely_polygon(cl.polygon_excluding_hull(), ax=ax, fc='r', ec='k')
    ax.plot(*p, 'ko')
    ax.plot(*pg, 'k*')
    if xlim is not None:
        ax.set_xlim(xlim)
    if ylim is not None:
        ax.set_ylim(ylim)
    return fig, ax


def draw_adm_ker(clusters, p, pg, xlim=None, ylim=None):
    valid_adm_ker_cls = [cl for cl in clusters if not (cl.admissible_kernel is None or cl.admissible_kernel.is_empty)]
    n_col = int(np.sqrt(len(valid_adm_ker_cls))) + 1
    fig, axs = plt.subplots(n_col, n_col)
    for i, cl in enumerate(valid_adm_ker_cls):
        ax_i = axs[i//n_col, i%n_col]
        [o.draw(ax=ax_i, show_name=1, show_reference=0, fc='lightgrey', ec='k', alpha=0.8) for o in cl.obstacles]
        if not cl.admissible_kernel.geom_type == 'Point':
            draw_shapely_polygon(cl.admissible_kernel, ax=ax_i, fc='y', alpha=0.3)
            if cl.kernel_points is not None:
                draw_shapely_polygon(shapely.geometry.Polygon(cl.kernel_points), ax=ax_i, fc='g', alpha=0.6)
        ax_i.plot(*p, 'ko')
        ax_i.plot(*pg, 'k*')
        if xlim is not None:
            ax_i.set_xlim(xlim)
        if ylim is not None:
            ax_i.set_ylim(ylim)
    return fig, axs
