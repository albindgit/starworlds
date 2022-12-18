import numpy as np
import matplotlib.pyplot as plt
from utils import Cone, draw_shapely_polygon


def test_affine_transformation():
    pass


def test_point_in_triangle():
    pass


def test_orientation():
    pass


def test_equilateral_triangle():
    pass


def test_convex_hull():
    pass


def test_generate_convex_polygon():
    pass


def test_generate_star_polygon():
    pass


def test_cone():
    n_cones_sqrt = 4
    n_cones_same_apex = 4
    n_points = 8
    n_several_cone_intersection = 2
    boxlim = [-1, 1]

    fig_int_sev, axs_int_sev = plt.subplots(n_cones_sqrt, n_cones_sqrt)
    fig_int, axs_int = plt.subplots(n_cones_sqrt, n_cones_sqrt)
    fig_p, axs_p = plt.subplots(n_cones_sqrt, n_cones_sqrt)
    for i in range(n_cones_sqrt ** 2):
        # Test cone intersection
        c_params = np.random.uniform(*boxlim, (2, 3))
        c1 = Cone(c_params[:, 0], c_params[:, 1] - c_params[:, 0], c_params[:, 2] - c_params[:, 0])
        c2_params = np.random.uniform(*boxlim, (2, 3))
        c2_apex = c_params[:, 0] if i < n_cones_same_apex else c2_params[:, 0]
        c2 = Cone(c2_apex, c2_params[:, 1] - c2_apex, c2_params[:, 2] - c2_apex)
        c_several_intersect = [c1, c2]
        for j in range(1, n_several_cone_intersection):
            c3_params = np.random.uniform(*boxlim, (2, 3))
            c3_apex = c_params[:, 0] if i < n_cones_same_apex else c3_params[:, 0]
            c_several_intersect += [Cone(c3_apex, c3_params[:, 1] - c3_apex, c3_params[:, 2] - c3_apex)]

        two_cones_intersection = c1.intersection(c2)
        several_cones_intersection = Cone.list_intersection(c_several_intersect, same_apex=i<n_cones_same_apex)

        # Two cones intersection plot
        fig_int.suptitle("Intersection of two cones")
        ax_int_i = axs_int[i//n_cones_sqrt, i%n_cones_sqrt]
        c1.draw(ax=ax_int_i, color='b', alpha=0.2, ray_color='b')
        c2.draw(ax=ax_int_i, color='r', alpha=0.2, ray_color='r')
        if not two_cones_intersection.is_empty and not two_cones_intersection.geom_type == 'Point':
            draw_shapely_polygon(two_cones_intersection, ax=ax_int_i, color='k', alpha=0.2, hatch='///')
        ax_int_i.set_xlim(1.2 * np.array(boxlim))
        ax_int_i.set_ylim(1.2 * np.array(boxlim))

        # Several cones intersection plot
        fig_int_sev.suptitle("Intersection of several cones")
        color = plt.cm.rainbow(np.linspace(0, 1, 2*n_several_cone_intersection))
        ax_int_sev_i = axs_int_sev[i//n_cones_sqrt, i%n_cones_sqrt]
        for j, c in enumerate(c_several_intersect):
            c.draw(ax=ax_int_sev_i, color=color[j], alpha=0.2, ray_color=color[j])
        if not several_cones_intersection.is_empty:
            draw_shapely_polygon(several_cones_intersection, ax=ax_int_sev_i, color='k', alpha=0.2, hatch='///')
        ax_int_sev_i.set_xlim(1.2 * np.array(boxlim))
        ax_int_sev_i.set_ylim(1.2 * np.array(boxlim))

        # Test point in cone
        fig_p.suptitle("Point in cone")
        ax_p_i = axs_p[i//n_cones_sqrt, i%n_cones_sqrt]
        xs = np.linspace(*boxlim, n_points)
        ys = np.linspace(*boxlim, n_points)
        XS, YS = np.meshgrid(xs, ys)
        c1.draw(ax=ax_p_i, color='b', alpha=0.2, ray_color='b')
        for j in range(n_points):
            for k in range(n_points):
                col = 'g' if c1.point_in_cone([XS[j, k], YS[j, k]]) else 'r'
                ax_p_i.plot(XS[j, k], YS[j, k], marker='.', color=col)
        ax_p_i.set_xlim(1.2 * np.array(boxlim))
        ax_p_i.set_ylim(1.2 * np.array(boxlim))


if (__name__) == "__main__":
    test_convex_hull()
    test_orientation()
    test_affine_transformation()
    test_equilateral_triangle()
    test_orientation()
    test_generate_convex_polygon()
    test_generate_star_polygon()
    test_point_in_triangle()
    test_cone()
    plt.show()
