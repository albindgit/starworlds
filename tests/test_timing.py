import numpy as np
import matplotlib.pyplot as plt
from obstacles import Ellipse, StarshapedPolygon
from obstacles import motion_model as mm
from utils import generate_convex_polygon
from starshaped_hull import cluster_and_starify, draw_clustering, draw_adm_ker, draw_star_hull
import shapely


def test_cluster_and_starify_compute():
    par = {'N_samples': 1000, 'pol_Nvert': 10, 'ell_n_pol': 30, 'ell_fraction': 0.5, 'pol_box': [2, 2],
           'ell_radius_mean': 1., 'ell_radius_std': 0.2, 'target_scene_coverage': 0.25,
           'No_min': 5, 'No_max': 50, 'rd_seed': 0, 'epsilon': 0.1}

    plot_fail = 0
    #  ----- Data generation  ------ #
    np.random.seed(par['rd_seed'])
    # Result data
    res = {'compute_time': np.zeros((par['N_samples'], 4)), 'n_iter': np.zeros(par['N_samples'], dtype=np.int32),
           'No': np.zeros(par['N_samples'], dtype=np.int32), 'Ncl': np.zeros(par['N_samples'], dtype=np.int32),
           'obstacle_area': np.zeros(par['N_samples']), 'obstacle_coverage': np.zeros(par['N_samples']),
           'scene_width': np.zeros(par['N_samples']), 'scene_coverage': np.zeros(par['N_samples'])}

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

    for i in range(par['N_samples']):
        # Generate obstacles
        res['No'][i] = np.random.randint(par['No_min'], par['No_max'] + 1)
        Nell = int(res['No'][i] * par['ell_fraction'])
        Npol = res['No'][i] - Nell
        obstacles = [
            Ellipse(a=np.random.normal(par['ell_radius_mean'], par['ell_radius_std'], 2), n_pol=par['ell_n_pol'])
            for j in range(Nell)]
        obstacles += [StarshapedPolygon(generate_convex_polygon(par['pol_Nvert'], par['pol_box']), xr=[0, 0],
                                           is_convex=True) for j in range(Npol)]

        # Compute area data
        res['obstacle_area'][i] = sum([o.area() for o in obstacles])

        # Setup scene
        res['scene_width'][i] = np.sqrt(res['obstacle_area'][i] / par['target_scene_coverage'] * 0.85)

        for j in range(Nell):
            # obstacles[j].set_motion_model(ob.Static(random_scene_point(res['scene_width'][i])))
            obstacles[j].set_motion_model(mm.Static(
                random_scene_point(res['scene_width'][i] - 2 * par['ell_radius_mean']) + par['ell_radius_mean']))
        for j in range(Npol):
            # obstacles[Nell+j].set_motion_model(ob.Static(random_scene_point(res['scene_width'][i])))
            obstacles[Nell + j].set_motion_model(
                mm.Static(random_scene_point(res['scene_width'][i] - par['pol_box'][0]) + par['pol_box'][0] / 2))

        # Compute coverage data
        res['obstacle_coverage'][i] = shapely.ops.unary_union([o.polygon() for o in obstacles]).area
        res['scene_coverage'][i] = res['obstacle_coverage'][i] / (res['scene_width'][i] ** 2)

        flag = 0
        n_failures = -1
        while flag == 0:
            n_failures += 1
            # Select collision free robot and goal positions
            x, xg = select_x_xg(res['scene_width'][i], obstacles)
            # Cluster and starify
            clusters, timing, flag, res['n_iter'][i] = cluster_and_starify(obstacles, x, xg, par['epsilon'],
                                                                           exclude_obstacles=0, make_convex=0,
                                                                           max_iterations=100,
                                                                           verbose=0)

            star_obstacles = [cl.cluster_obstacle for cl in clusters]
            res['compute_time'][i, :] = timing
            res['Ncl'][i] = len(clusters)

            if plot_fail and flag == 0:
                _, ax_i = plt.subplots()
                [o.draw(ax=ax_i, fc='g', alpha=0.8) for o in star_obstacles]
                [o.draw(ax=ax_i, show_name=1, show_reference=0, ec='k', linestyle='--') for o in obstacles]
                ax_i.plot(*x, 'rx', markersize=16)
                ax_i.plot(*xg, 'rx', markersize=16)
                ax_i.set_xlim([0, res['scene_width'][i]])
                ax_i.set_ylim([0, res['scene_width'][i]])
                if flag == 0:
                    ax_i.set_title(
                        'Fail\nSample: {}, #O: {}, #Cl: {}, Time: {:.1f}, It: {}, Scene coverage: {:.2f}({:.2f})'.format(
                            i, res['No'][i], res['Ncl'][i], sum(res['compute_time'][i, :]), res['n_iter'][i],
                            res['scene_coverage'][i], par['target_scene_coverage']))
                else:
                    ax_i.set_title(
                        'Sample: {}, #O: {}, #Cl: {}, Time: {:.1f}, It: {}, Scene coverage: {:.2f}({:.2f})'.format(
                            i, res['No'][i], res['Ncl'][i], sum(res['compute_time'][i, :]), res['n_iter'][i],
                            res['scene_coverage'][i], par['target_scene_coverage']))
                plt.show()

            if n_failures == 5:
                break

    #  ----- Postprocessing ------ #
    total_compute_time = np.sum(res['compute_time'], axis=1)
    binc_niter = np.bincount(res['n_iter'])[min(res['n_iter']):]
    print(np.vstack((np.arange(min(res['n_iter']), min(res['n_iter']) + len(binc_niter)), binc_niter)))

    cl = ['r', 'g', 'b', 'y', 'k', 'c']
    mk = ['o', '+', '^', 'x', 'd', '8']
    tct_it = [None] * len(binc_niter)
    No_it = [None] * len(binc_niter)
    ct_it = [[]] * len(binc_niter)
    scene_coverage_it = [None] * len(binc_niter)
    for i in range(len(binc_niter)):
        scene_coverage_it[i] = np.ma.masked_where(res['n_iter'] != min(res['n_iter']) + i, res['scene_coverage'])
        tct_it[i] = np.ma.masked_where(res['n_iter'] != min(res['n_iter']) + i, total_compute_time)
        No_it[i] = np.ma.masked_where(res['n_iter'] != min(res['n_iter']) + i, res['No'])
        for j in range(3):
            ma = np.ma.masked_where(res['n_iter'] != min(res['n_iter']) + i, res['compute_time'][:, j])
            ct_it[i] += [ma.data]

    timename = ['Clustering', 'Adm Ker', 'St Hull']
    if max(res['No']) - min(res['No']) > 0:
        _, axs = plt.subplots(3)
        for j in range(3):
            for i in range(len(binc_niter)):
                axs[j].scatter(No_it[i], ct_it[i][j], c=cl[i], marker=mk[i])
            axs[j].set_xlabel('No'), axs[j].set_ylabel('Time [ms]')
            axs[j].set_title(timename[j])
    plt.tight_layout()

    _, axs = plt.subplots(1, 2)
    # ax.scatter(res['No'], total_compute_time)
    for i in range(len(binc_niter)):
        axs[0].scatter(No_it[i], tct_it[i], c=cl[i], marker=mk[i], s=np.square(scene_coverage_it[i] * 20))
        axs[1].scatter(scene_coverage_it[i], tct_it[i], c=cl[i], marker=mk[i], s=No_it[i] * 2)

    _, ax = plt.subplots()
    ax.scatter(res['No'], np.divide(res['compute_time'][:, 0], total_compute_time), color='r')
    ax.scatter(res['No'], np.divide(res['compute_time'][:, 1], total_compute_time), color='g')
    ax.scatter(res['No'], np.divide(res['compute_time'][:, 2], total_compute_time), color='b')
    ax.set_title("Clustering fraction of total"), ax.set_xlabel('No'), ax.set_ylabel('Cl time / Tot time')

    _, axs = plt.subplots(2, 2)
    axs[0, 0].scatter(res['No'], res['obstacle_area'])
    axs[0, 0].scatter(res['No'], res['obstacle_coverage']), axs[0, 0].set_title('Obstacle area')
    axs[0, 0].set_xlabel('#Obstacles'), axs[1, 0].set_ylabel('Obstacle area')
    axs[1, 0].scatter(res['scene_coverage'], res['n_iter']), axs[1, 0].set_title('#Iterations')
    axs[1, 0].set_xlabel('Scene coverage'), axs[1, 0].set_ylabel('#Iterations')
    axs[0, 1].scatter(res['No'], np.divide(res['Ncl'], res['No'])), axs[0, 1].set_title('#Clusters')
    axs[0, 1].set_xlabel('#Obstacles'), axs[0, 1].set_ylabel('#Clusters/#Obstacles')
    axs[1, 1].scatter(res['scene_coverage'], np.divide(res['Ncl'], res['No'])), axs[1, 1].set_title('#Clusters/#No')
    axs[1, 1].set_xlabel('Scene coverage'), axs[1, 1].set_ylabel('#Clusters/#Obstacles')
    plt.tight_layout()

    scene_coverage = res['scene_coverage'] * 100
    cov_span = max(scene_coverage) - min(scene_coverage)
    binwidth = 1
    _, ax = plt.subplots()
    ax.hist(scene_coverage, bins=int(cov_span / binwidth),
            color='blue', edgecolor='black')
    ymin, ymax = ax.get_ylim()
    ax.plot([par['target_scene_coverage'] * 100, par['target_scene_coverage'] * 100], [ymin, ymax], 'r--')
    ax.set_title('Histogram with Binwidth = %d' % binwidth, size=30)
    ax.set_xlabel('Coverage', size=22)
    ax.set_ylabel('Samples', size=22)


if (__name__) == "__main__":
    test_cluster_and_starify_compute()
    plt.show()