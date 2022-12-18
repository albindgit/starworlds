import numpy as np
import matplotlib.pyplot as plt
from starshaped_hull import admissible_kernel
from utils import draw_shapely_polygon, Cone
from obstacles import Polygon, Ellipse
from obstacles import motion_model as mm

def test_admissible_kernel():
    pol = Polygon([[0, 0], [1, 0], [2, 2], [2, 4], [-0.75, 1], [-1, 4], [-2, 2], [-2, 1], [-1.5, 0], [-2, -1]])

    x_exclude = np.array([0, 2])

    ad_ker = admissible_kernel(pol, x_exclude)

    _, ax = pol.draw()
    draw_shapely_polygon(ad_ker.polygon(), ax=ax, fc='y', alpha=0.6)
    ax.plot(x_exclude[0], x_exclude[1], 'ro')
    ax.set_xlim([-3, 6])
    ax.set_ylim([-1, 5])


def test_adm_ker_ellipse():
    ell = Ellipse([1.08877265, 1.06673487], motion_model=mm.Static([4.61242385, 1.87941425]))
    p = np.array([3.641344, 4.87955125])
    pg = [0.73012219, 8.95180958]
    adm_ker_p = admissible_kernel(ell, p)# Cone.list_intersection([adm_ker_robot_cones[o.id()] for o in cl.obstacles])
    adm_ker_pg = admissible_kernel(ell, pg)
    ad_ker = adm_ker_p.intersection(adm_ker_pg)

    _, ax = ell.draw()
    draw_shapely_polygon(ad_ker, ax=ax, fc='y', alpha=0.6)
    draw_shapely_polygon(adm_ker_p.polygon(), ax=ax, fc='None', ec='k')
    draw_shapely_polygon(adm_ker_pg.polygon(), ax=ax, fc='None', ec='k')
    ax.plot(*p, 'rx')
    ax.plot(*pg, 'rx')
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)


def test_kernel_starshaped_hull_single():
    pass


def test_kernel_starshaped_hull_cluster():
    pass


if (__name__) == "__main__":
    test_admissible_kernel()
    test_adm_ker_ellipse()
    test_kernel_starshaped_hull_single()
    test_kernel_starshaped_hull_cluster()
    plt.show()