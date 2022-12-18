import shapely
import numpy as np
from obstacles import Frame, StarshapedPrimitiveCombination, Polygon, StarshapedPolygon
from utils import is_cw, is_ccw, is_collinear, equilateral_triangle, Cone, convex_hull, \
    tic, toc, draw_shapely_polygon
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

    t0 = tic()
    # Use polygon approximations for intersection check
    cluster_polygons = [cl.polygon() for cl in clusters]
    t1 = toc(t0)
    # print("Polygon: {:.2f}".format(t1))
    t0 = tic()

    # Find intersections
    intersections_exist = False
    for i in range(No):
        intersection_idcs += [[i]]
        for j in range(i + 1, No):
            if cluster_polygons[i].intersects(cluster_polygons[j]):
                intersection_idcs[i] += [j]
                intersections_exist = True

    t1 = toc(t0)
    # print("Intersect ident: {:.2f}".format(t1))
    t0 = tic()

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

    t1 = toc(t0)
    # print("Clustering: {:.2f}".format(t1))
    t0 = tic()

    # Create obstacle clusters
    cluster_obstacles = [cl.obstacles for cl in clusters]
    # cluster_obstacles = [cl.obstacles if isinstance(cl, ObstacleCluster) else [cl] for cl in clusters]
    new_clusters = []
    for i in range(No):
        if intersection_idcs[i]:
            new_clusters += [ObstacleCluster([o for j in intersection_idcs[i] for o in cluster_obstacles[j]])]

    t1 = toc(t0)
    # print("Creation: {:.2f}".format(t1))

    return new_clusters, intersections_exist


def compute_kernel_points(cl, x, xg, epsilon, cl_prev):
    if len(cl.obstacles) == 1 and cl.obstacles[0].is_starshaped():
        triangle_center = cl.obstacles[0].xr(Frame.GLOBAL)
    else:
        triangle_center_prev = np.mean(cl_prev.kernel_points, axis=0) if cl_prev else None

        triangle_center_selection_set = cl.admissible_kernel.intersection(cl.polygon_excluding_hull())
        # triangle_center_selection_set = triangle_center_selection_set.buffer(-epsilon/2)
        if triangle_center_selection_set.is_empty:
            triangle_center_selection_set = cl.admissible_kernel

        if triangle_center_prev is None:
            tc, _ = shapely.ops.nearest_points(cl.admissible_kernel, triangle_center_selection_set.centroid)
            triangle_center = np.array(tc.coords[0])
            if is_collinear(x, xg, triangle_center):
                triangle_center += 1e-4 * np.ones(2)
        # Use previous triangle center if still in selection set
        elif triangle_center_selection_set.contains(shapely.geometry.Point(triangle_center_prev))\
                and not is_collinear(x, xg, triangle_center_prev):
            triangle_center = triangle_center_prev
        else:
            x_xg_line = shapely.geometry.LineString([x, xg])
            splitted_tcss = shapely.ops.split(triangle_center_selection_set, x_xg_line).geoms
            triangle_center_selection_set = splitted_tcss[0]
            if len(splitted_tcss) > 1:
                for i in range(1, len(splitted_tcss)):
                    if is_ccw(x, xg, splitted_tcss[i].centroid.coords[0]) == is_ccw(x, xg, triangle_center_prev):
                        triangle_center_selection_set = splitted_tcss[i]
                        break
            tc, _ = shapely.ops.nearest_points(cl.admissible_kernel, triangle_center_selection_set.centroid)
            triangle_center = np.array(tc.coords[0])

        if cl.admissible_kernel.geom_type == 'Polygon':
            dist = cl.admissible_kernel.exterior.distance(shapely.geometry.Point(triangle_center))
        else:
            tc = shapely.geometry.Point(triangle_center)
            dist = min([p.exterior.distance(tc) for p in cl.admissible_kernel.geoms])
        triangle_length = min(epsilon, 0.9 * dist)
        kernel_points = equilateral_triangle(triangle_center, triangle_length)
        return kernel_points

    #     triangle_center_prev = np.mean(cl_prev.kernel_points, axis=0) if cl_prev else None
    #
    #     t0 = tic()
    #     triangle_center_selection_set = cl.admissible_kernel.intersection(cl.polygon_excluding_hull())
    #     # triangle_center_selection_set = triangle_center_selection_set.buffer(-epsilon/2)
    #     if triangle_center_selection_set.is_empty:
    #         triangle_center_selection_set = cl.admissible_kernel
    #     tcss_time = toc(t0)
    #
    #     t0 = tic()
    #     # Use previous triangle center if still in admissible kernel
    #     if triangle_center_prev is not None and triangle_center_selection_set.contains(shapely.geometry.Point(triangle_center_prev))\
    #             and not is_collinear(x, xg, triangle_center_prev):
    #         triangle_center = triangle_center_prev
    #         # print("prev")
    #     else:
    #         # Else select triangle center based on line l(x,xg)
    #         if triangle_center_prev is not None:
    #             x_xg_line = shapely.geometry.LineString([x, xg])
    #             splitted_tcss = shapely.ops.split(triangle_center_selection_set, x_xg_line).geoms
    #             triangle_center_selection_set = splitted_tcss[0]
    #             if len(splitted_tcss) > 1:
    #                 # If previous triangle center exists, select triangle center to be on same side of line l(x,xg) as this
    #                 if triangle_center_prev is not None:
    #                     for i in range(1, len(splitted_tcss)):
    #                         if is_ccw(x, xg, splitted_tcss[i].centroid.coords[0]) == is_ccw(x, xg, triangle_center_prev):
    #                             triangle_center_selection_set = splitted_tcss[i]
    #                             break
    #
    #         tc, _ = shapely.ops.nearest_points(cl.admissible_kernel, triangle_center_selection_set.centroid)
    #         triangle_center = np.array(tc.coords[0])
    #
    #     tc_time = toc(t0)
    #
    #
    # t0 = tic()
    # if cl.admissible_kernel.geom_type == 'Polygon':
    #     dist = cl.admissible_kernel.exterior.distance(shapely.geometry.Point(triangle_center))
    # else:
    #     tc = shapely.geometry.Point(triangle_center)
    #     dist = min([p.exterior.distance(tc) for p in cl.admissible_kernel.geoms])
    # triangle_length = min(epsilon, 0.9 * dist)
    # kernel_points = equilateral_triangle(triangle_center, triangle_length)
    #
    # # length = epsilon
    # # kernel_points = equilateral_triangle(triangle_center, length)
    # # # If triangle too big, reduce size until fully contained in kernel_selection_set
    # # it = 0
    # # it_warn = 2
    # # while not cl.admissible_kernel.contains(shapely.geometry.Polygon(kernel_points)):
    # #     it += 1
    # #     dist = cl.admissible_kernel.exterior.distance(shapely.geometry.Point(triangle_center))
    # #     kernel_points = equilateral_triangle(triangle_center, 0.9*dist)
    # #     # length = 0.5 * length
    # #     # kernel_points = equilateral_triangle(triangle_center, length)
    # #     if it > it_warn-1 and it < it_warn+1:
    # #         print("[KERNEL POINT SELECTION]: WARNING: many iterations")
    # kp_time = toc(t0)
    #
    # # print("(TSS/TC/KP) calculation ({:.2f}/{:.2f}/{:.2f})".format(tcss_time, tc_time, kp_time))
    #
    # return kernel_points


# Input: Convex obstacles, excluding points x and xg, kernel width epsilon
def cluster_and_starify(obstacles, x, xg, epsilon, max_compute_time=np.inf, previous_clusters=None,
                        make_convex=False, exclude_obstacles=False, max_iterations=np.inf, verbose=False,
                        timing_verbose=False, return_history=False):

    t0 = tic()

    # Exit flags
    INCOMPLETE = 0
    COMPLETE = 1
    MAX_COMPUTE_TIME_CONVEX_HULL = 2

    cluster_iterations = []

    def default_result(n_iter):
        nonlocal cluster_iterations
        if return_history:
            if n_iter == 0:
                cluster_iterations = [ObstacleCluster([o]) for o in obstacles]
            return [ObstacleCluster([o]) for o in obstacles], [0.] * 4, INCOMPLETE, n_iter, cluster_iterations
        else:
            return [ObstacleCluster([o]) for o in obstacles], [0.] * 4, INCOMPLETE, n_iter

    # Variable initialization
    cluster_time, kernel_time, hull_time, convex_time = [[]] * 4
    adm_ker_robot_cones = {obs.id(): None for obs in obstacles}
    adm_ker_goal_cones = {obs.id(): None for obs in obstacles}
    adm_ker_obstacles = {obs.id(): {} for obs in obstacles}

    # Compute admissible kernel for all obstacles
    for o in obstacles:
        if max_compute_time < toc(t0):
            return default_result(0)
        # Find admissible kernel
        adm_ker_robot_cones[o.id()] = sh.admissible_kernel(o, x)
        adm_ker_goal_cones[o.id()] = sh.admissible_kernel(o, xg)

        if adm_ker_robot_cones[o.id()] is None:
            if verbose:
                print("[Cluster and Starify]: Robot position is not a free exterior point of obstacle " + str(o.id()))
                import matplotlib.pyplot as plt
                _, ax = o.draw()
                ax.plot(*x, 'kx')
                ax.set_xlim([-1, 11])
                ax.set_ylim([-1, 11])

                tps = o.tangent_points(x)
                print(tps)

                plt.show()
            return default_result(0)

        if adm_ker_goal_cones[o.id()] is None:
            if verbose:
                print("[Cluster and Starify]: Goal position is not a free exterior point of obstacle " + str(o.id()))
            return default_result(0)

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
    # Initialize clusters as single obstacles
    clusters = []
    kernel_time, hull_time, cluster_time = [0.], [0.], [0.]
    for o in obstacles:
        # Ensure not xr in l(x,xg)
        while is_collinear(x, o.xr(), xg):
            o.set_xr(o.xr(Frame=Frame.OBSTACLE) + np.random.normal(0, 0.01, 2))
        clusters += [ObstacleCluster([o])]
        cl = clusters[-1]
        adm_ker_robot = Cone.list_intersection([adm_ker_robot_cones[o.id()] for o in cl.obstacles])
        adm_ker_goal = Cone.list_intersection([adm_ker_goal_cones[o.id()] for o in cl.obstacles])
        cl.admissible_kernel = adm_ker_robot.intersection(adm_ker_goal)

        if not o.is_starshaped():
            cl = clusters[-1]
            t1 = tic()
            cl.admissible_kernel = adm_ker_robot_cones[o.id()].polygon().intersection(adm_ker_goal_cones[o.id()].polygon())
            kernel_time[0] += toc(t1)

            t1 = tic()
            cl_prev = None
            if previous_clusters:
                for p_cl in previous_clusters:
                    if cl.name == p_cl.name:
                        cl_prev = p_cl
            cl.kernel_points = compute_kernel_points(cl, x, xg, epsilon, cl_prev, dx_prev)
            # -- Compute starshaped hull of cluster
            cluster_hull_extensions = sh.kernel_starshaped_hull(o, cl.kernel_points)
            k_centroid = np.mean(cl.kernel_points, axis=0)
            pols = [o.polygon(), cluster_hull_extensions]
            cl._polygon = shapely.ops.unary_union(pols)
            cl.cluster_obstacle = StarshapedPolygon(cl._polygon, xr=k_centroid, id=o.id())
            hull_time[0] += toc(t1)

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

    ker_sel_time, hull_compute_time, star_obj_time = [0], [0], [0]
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
            return default_result(n_iter)

        # Find starshaped obstacle representation for each cluster
        for i, cl in enumerate(clusters):
            # If cluster already calculated keep it
            if cl.name in cluster_history:
                clusters[i] = cluster_history[cl.name]
                continue

            # ----------- Admissible Kernel ----------- #
            t1 = tic()

            # If cluster is two convex obstacles
            if len(cl.obstacles) == 2 and cl.obstacles[0].is_convex() and cl.obstacles[1].is_convex():
                cl.admissible_kernel = cl.obstacles[0].polygon().intersection(cl.obstacles[1].polygon()).buffer(0.01)
            else:
                adm_ker_robot = Cone.list_intersection([adm_ker_robot_cones[o.id()] for o in cl.obstacles], same_apex=True)
                adm_ker_goal = Cone.list_intersection([adm_ker_goal_cones[o.id()] for o in cl.obstacles], same_apex=True)
                cl.admissible_kernel = adm_ker_robot.intersection(adm_ker_goal)
                cl.admissible_kernel = adm_ker_goal

                if cl.admissible_kernel.is_empty or cl.admissible_kernel.area < 1e-6:
                    if verbose:
                        print("[Cluster and Starify]: Could not find disjoint starshaped obstacles. Admissible kernel empty for the cluster " + cl.name)
                    cluster_iterations += [clusters]
                    return default_result(n_iter)

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
            cl.kernel_points = compute_kernel_points(cl, x, xg, epsilon, cl_prev)
            ker_sel_time[n_iter] += toc(t1)
            t1 = tic()

            # -- Compute starshaped hull of cluster
            cl_id = "new" if cl_prev is None else cl_prev.cluster_obstacle.id()
            cluster_hull_extensions = sh.kernel_starshaped_hull(cl.obstacles, cl.kernel_points, timing_verbose=timing_verbose)
            hull_compute_time[n_iter] += toc(t1)

            t1 = tic()
            k_centroid = np.mean(cl.kernel_points, axis=0)
            if False and all([isinstance(o, Polygon) for o in cl.obstacles]):
                pols = [o.polygon() for o in cl.obstacles]
                if cluster_hull_extensions:
                    pols += [cluster_hull_extensions]
                cl._polygon = shapely.ops.unary_union(pols)
                cl.cluster_obstacle = StarshapedPolygon(cl._polygon, xr=k_centroid, id=cl_id)
            else:
                # Non-starshaped polygons are included in the cluster hull
                cl_obstacles = [o for o in cl.obstacles if o.is_starshaped()]
                cl.cluster_obstacle = StarshapedPrimitiveCombination(cl_obstacles, cluster_hull_extensions, xr=k_centroid, id=cl_id)
                cl.polygon()

            star_obj_time[n_iter] += toc(t1)
            # ----------- End Starshaped Hull ----------- #

            # -- Add cluster to history
            cluster_history[cl.name] = cl

        hull_time[n_iter] = ker_sel_time[n_iter] + hull_compute_time[n_iter] + star_obj_time[n_iter]

        if not n_iter+1 < max_iterations:
            break

        if return_history:
            cluster_iterations += [clusters]

        # ----------- Clustering ----------- #
        t1 = tic()
        # -- Cluster star obstacles such that no intersection exists
        clusters, intersection_exists = get_intersection_clusters(clusters)
        cluster_time[n_iter] = toc(t1)
        # ----------- End Clustering ----------- #

        n_iter += 1

    # ----------- Make Convex ----------- #

    # TODO: Check if computational efficiency can be improved
    if make_convex:
        # Make convex if no intersection occurs
        t1 = tic()
        for j, cl in enumerate(clusters):
            if not cl.cluster_obstacle.is_convex():
                # Check compute time
                if max_compute_time < toc(t0):
                    if verbose:
                        print("[Cluster and Starify]: Max compute time in convex hull.")
                    return clusters, [cluster_time, kernel_time, hull_time, toc(t1)], MAX_COMPUTE_TIME_CONVEX_HULL, n_iter

                v = np.array(cl.polygon().exterior.coords)[:-1, :]
                hull = ConvexHull(v)
                hull_polygon = shapely.geometry.Polygon(v[hull.vertices, :])
                # hull_polygon = shapely.geometry.Polygon(convex_hull(v))
                if not any([hull_polygon.contains(shapely.geometry.Point(x_ex)) for x_ex in [x, xg]]) and not any(
                        [hull_polygon.intersects(clusters[k].polygon()) for k in range(len(clusters)) if k != j]):
                    clusters[j].cluster_obstacle = StarshapedPrimitiveCombination(cl.obstacles, hull_polygon, cl.cluster_obstacle.xr(Frame.GLOBAL))

                # conv_obs = StarshapedPolygon(v[hull.vertices, :], xr=cl.cluster_obstacle.xr(Frame.GLOBAL), id=cl.cluster_obstacle.id())
                # if all([conv_obs.exterior_point(x_ex) for x_ex in [x, xg]]) and not any(
                #         [conv_obs.intersects(clusters[k].cluster_obstacle) for k in range(len(clusters)) if k != j]):
                #     # clusters[j].cluster_obstacle = StarshapedHullPrimitiveCombination(cl.obstacles, cl.obstacles + [conv_obs], cl.cluster_obstacle.xr(Frame.GLOBAL))
                #     clusters[j].cluster_obstacle = StarshapedPrimitiveCombination(cl.obstacles, [conv_obs], cl.cluster_obstacle.xr(Frame.GLOBAL))
                    # clusters[j].cluster_obstacle = conv_obs

        convex_time += [toc(t1)]
    # ----------- End Make Convex ----------- #

    # Ensure convergence of SOADS by having xr not in l(x,xg)
    # for cl in clusters:
    #     while is_collinear(x, cl.cluster_obstacle.xr(), xg):
    #         # Check compute time
    #         if max_compute_time < toc(t0):
    #             if verbose:
    #                 print("[Cluster and Starify]: Max compute time in xr adjustment.")
    #             return clusters, [cluster_time, kernel_time, hull_time, convex_time], MAX_COMPUTE_TIME_XR_ADJUSTMENT
    #         if cl.kernel_points is not None:
    #             cl.cluster_obstacle.set_xr(point_in_triangle(*cl.kernel_points), Frame.GLOBAL)
    #         else:
    #             cl.cluster_obstacle.set_xr(cl.cluster_obstacle.xr()+np.random.normal(0, 0.01, 2), Frame.GLOBAL)
    #         # cl.cluster_obstacle.set_xr(new_xr, Frame.GLOBAL)


    if timing_verbose:
        print("------------\nInit kernel timing: {:.1f}".format(init_kernel_time))
        for i in range(n_iter):
            print("Iteration {} timing (Cluster/AdmKer/StHull): [{:.1f},{:.1f},{:.1f}]".format(i, cluster_time[i], kernel_time[i], hull_time[i]))
            print("\t Hull timing divide: (KerSel/Hull/Obj) calculation [{:.1f}/{:.1f}/{:.1f}]".format(ker_sel_time[i], hull_compute_time[i], star_obj_time[i]))

    if return_history:
        return clusters, [sum(cluster_time), init_kernel_time+sum(kernel_time), sum(hull_time), sum(convex_time)], COMPLETE, n_iter, cluster_iterations
    else:
        return clusters, [sum(cluster_time), init_kernel_time+sum(kernel_time), sum(hull_time), sum(convex_time)], COMPLETE, n_iter


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
