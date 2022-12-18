import numpy as np
import matplotlib.pyplot as plt
from obstacles import Ellipse, StarshapedPolygon
from obstacles import motion_model as mm
from utils import generate_convex_polygon, draw_shapely_polygon
from starshaped_hull import cluster_and_starify, draw_clustering, draw_adm_ker, draw_star_hull
import shapely


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
    obstacles = [Ellipse(a=np.random.normal(ell_radius_mean, ell_radius_std, 2), n_pol=10) for j in range(Nell)]
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
    clusters, timing, flag, n_iter, cluster_history = cluster_and_starify(obstacles, x, xg, epsilon, make_convex=0,
                                                                          verbose=1, return_history=1, timing_verbose=1)

    # Draw iteration steps
    for i, clusters_i in enumerate(cluster_history[1:]):
        _, ax = draw_clustering(clusters_i, x, xg, xlim=[0, scene_width], ylim=[0, scene_width])
        ax.set_title("Clustering, Iteration {}/{}".format(i+1, len(cluster_history)))
        fig, axs = draw_adm_ker(clusters_i, x, xg, xlim=[-0.2*scene_width, 1.2*scene_width], ylim=[-0.2*scene_width, 1.2*scene_width])
        fig.suptitle("Admissible Kernel, Iteration {}/{}".format(i+1, len(cluster_history)))
        _, ax = draw_star_hull(clusters_i, x, xg, xlim=[0, scene_width], ylim=[0, scene_width])
        ax.set_title("Starshaped Hull, Iteration {}/{}".format(i+1, len(cluster_history)))


def test_moving_cluster():
    obstacles = [
        Ellipse([1, 1], motion_model=mm.Static([-1, 0.3])),
        Ellipse([1, 1], motion_model=mm.Static([0., 0.3])),
        Ellipse([1, 1], motion_model=mm.Static([1, 0.3]))
    ]
    p0 = np.array([0.1, -2.5])
    pg = np.array([0.1, 2.5])
    epsilon = 0.2
    xlim = [-3, 3]
    ylim = [-3, 3]

    clusters, timing, flag, n_iter = cluster_and_starify(obstacles, p0, pg, epsilon)

    fig, axs = plt.subplots(1, 3)
    [o.draw(ax=axs[0], show_reference=0, fc='lightgrey', ec='k', alpha=0.8) for o in obstacles]
    axs[0].plot(*p0, 'ko')
    axs[0].plot(*pg, 'k*')
    axs[0].plot(*clusters[0].cluster_obstacle.xr(), 'gd')
    axs[0].plot(*zip(p0, pg), '--')
    axs[0].set_xlim(xlim)
    axs[0].set_ylim(ylim)

    xr_prev = clusters[0].cluster_obstacle.xr()
    obstacles[0].set_motion_model(mm.Static([-1, -0.3]))
    obstacles[1].set_motion_model(mm.Static([0., -0.3]))
    obstacles[2].set_motion_model(mm.Static([1, -0.3]))
    clusters, timing, flag, n_iter = cluster_and_starify(obstacles, p0, pg, epsilon, previous_clusters=clusters)

    [o.draw(ax=axs[1], show_reference=0, fc='lightgrey', ec='k', alpha=0.8) for o in obstacles]
    axs[1].plot(*p0, 'ko')
    axs[1].plot(*pg, 'k*')
    axs[1].plot(*xr_prev, 'sk')
    axs[1].plot(*clusters[0].cluster_obstacle.xr(), 'gd')
    axs[1].plot(*zip(p0, pg), '--')
    axs[1].set_xlim(xlim)
    axs[1].set_ylim(ylim)

    xr_prev = clusters[0].cluster_obstacle.xr()
    obstacles[0].set_motion_model(mm.Static([-1, -1]))
    obstacles[1].set_motion_model(mm.Static([0., -1]))
    obstacles[2].set_motion_model(mm.Static([1, -1]))
    clusters, timing, flag, n_iter = cluster_and_starify(obstacles, p0, pg, epsilon, previous_clusters=clusters)

    x_xg_line = shapely.geometry.LineString([p0, pg])
    kernel_selection_set = shapely.ops.split(clusters[0].cluster_obstacle.polygon(), x_xg_line).geoms[0]

    [o.draw(ax=axs[2], show_reference=0, fc='lightgrey', ec='k', alpha=0.8) for o in obstacles]
    draw_shapely_polygon(kernel_selection_set, ax=axs[2], hatch='///', fill=False, linestyle='None')
    axs[2].plot(*p0, 'ko')
    axs[2].plot(*pg, 'k*')
    axs[2].plot(*xr_prev, 'sk')
    axs[2].plot(*clusters[0].cluster_obstacle.xr(), 'gd')
    axs[2].plot(*zip(p0, pg), '--')
    axs[2].set_xlim(xlim)
    axs[2].set_ylim(ylim)


if (__name__) == "__main__":
    test_cluster_and_starify()
    # test_moving_cluster()
    plt.show()