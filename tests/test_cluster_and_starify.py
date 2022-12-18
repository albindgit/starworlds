import numpy as np
import matplotlib.pyplot as plt
from obstacles import Ellipse, StarshapedPolygon
from obstacles import motion_model as mm
from utils import generate_convex_polygon
from starshaped_hull import cluster_and_starify, draw_clustering, draw_adm_ker, draw_star_hull


def test_get_intersection_clusters():
    pass


def test_compute_kernel_points():
    pass


def test_cluster_and_starify():
    n_obstacles = 40
    ellipse_fraction = .5
    ell_radius_mean, ell_radius_std = 1, 0.2
    n_vertices, pol_box = 8, [2, 2]
    target_scene_coverage = 0.3
    epsilon = 0.2

    np.random.seed(0)

    def random_scene_point(scene_width):
        return np.random.rand(2) * scene_width

    def select_x_xg(scene_width, obstacles):
        x = random_scene_point(scene_width)
        while any([o.interior_point(x) for o in obstacles]):
            x = random_scene_point(scene_width)
        xg = random_scene_point(scene_width)
        while any([o.interior_point(xg) for o in obstacles]):
            xg = random_scene_point(scene_width)
        return x, xg

    # Generate obstacles
    Nell = int(n_obstacles * ellipse_fraction)
    Npol = n_obstacles - Nell
    obstacles = [Ellipse(a=np.random.normal(ell_radius_mean, ell_radius_std, 2)) for j in range(Nell)]
    obstacles += [StarshapedPolygon(generate_convex_polygon(n_vertices, pol_box), xr=[0, 0], is_convex=True) for j in range(Npol)]

    # Compute area data
    obstacle_area = sum([o.area() for o in obstacles])

    # Setup scene
    scene_width = np.sqrt(obstacle_area / target_scene_coverage * 0.9)

    for j in range(Nell):
        # obstacles[j].set_motion_model(ob.Static(random_scene_point(res['scene_width'][i])))
        obstacles[j].set_motion_model(mm.Static(random_scene_point(scene_width - 2 * ell_radius_mean) + ell_radius_mean))
    for j in range(Npol):
        # obstacles[Nell+j].set_motion_model(ob.Static(random_scene_point(res['scene_width'][i])))
        obstacles[Nell + j].set_motion_model(mm.Static(random_scene_point(scene_width - pol_box[0]) + pol_box[0] / 2))
    [obs.polygon() for obs in obstacles]

    # Select collision free robot and goal positions
    x, xg = select_x_xg(scene_width, obstacles)
    # Cluster and starify
    clusters, timing, flag, n_iter, cluster_history = cluster_and_starify(obstacles, x, xg, epsilon, make_convex=1, verbose=1, return_history=1)

    print(timing)

    star_obstacles = [cl.cluster_obstacle for cl in clusters]

    if flag == 0:
        for i, clusters_i in enumerate(cluster_history):
            _, ax = draw_clustering(clusters_i, x, xg, xlim=[0, scene_width], ylim=[0, scene_width])
            ax.set_title("Clustering, Iteration {}/{}".format(i+1, len(cluster_history)))
            fig, axs = draw_adm_ker(clusters_i, x, xg, xlim=[0, scene_width], ylim=[0, scene_width])
            fig.suptitle("Admissible Kernel, Iteration {}/{}".format(i+1, len(cluster_history)))
            _, ax = draw_star_hull(clusters_i, x, xg, xlim=[0, scene_width], ylim=[0, scene_width])
            ax.set_title("Starshaped Hull, Iteration {}/{}".format(i+1, len(cluster_history)))
            plt.show()
    else:
        _, ax = plt.subplots()
        [o.draw(ax=ax, fc='g', alpha=0.8) for o in star_obstacles]
        [o.draw(ax=ax, show_name=0, show_reference=0, ec='k', linestyle='--') for o in obstacles]
        ax.plot(*x, 'rx', markersize=16)
        ax.plot(*xg, 'rx', markersize=16)
        ax.set_xlim([0, scene_width])
        ax.set_ylim([0, scene_width])

if (__name__) == "__main__":
    test_get_intersection_clusters()
    test_compute_kernel_points()
    test_cluster_and_starify()
    plt.show()