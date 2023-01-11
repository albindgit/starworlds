import numpy as np
import matplotlib.pyplot as plt
from obstacles import Ellipse, Polygon, StarshapedPolygon, StarshapedPrimitiveCombination, Frame
from obstacles import motion_model as mm
from utils import generate_convex_polygon, draw_shapely_polygon, generate_star_polygon
import shapely.geometry


def test_ellipse():
    ell_axes = [2, 1]
    ell_pos = [0, 0.5]
    xlim = [ell_pos[0] - 2 * ell_axes[0], ell_pos[0] + 2 * ell_axes[0]]
    ylim = [ell_pos[1] - 2 * ell_axes[1], ell_pos[1] + 2 * ell_axes[1]]
    ell = Ellipse(ell_axes, xr=[0, .9], motion_model=mm.Static(ell_pos, 1))
    while True:
        x = np.array([np.random.uniform(*xlim),np.random.uniform(*ylim)])
        if ell.exterior_point(x):
            break
    b = ell.boundary_mapping(x)
    n = ell.normal(x)
    tp = ell.tangent_points(x)
    dir = ell.reference_direction(x)

    _, ax = ell.draw()
    ax.plot(*zip(ell.xr(Frame.GLOBAL), x), 'k--o')
    if b is not None:
        ax.plot(*b, 'y+')
        ax.quiver(*b, *n)
    if tp:
        ax.plot(*zip(x, tp[0]), 'g:')
        ax.plot(*zip(x, tp[1]), 'g:')
    ax.quiver(*ell.xr(Frame.GLOBAL), *dir, color='c', zorder=3)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


def test_nonstar_polygon():
    pass


def test_star_polygon():
    avg_radius = 1
    xlim = [-2*avg_radius, 2*avg_radius]
    ylim = xlim
    pol = StarshapedPolygon(generate_star_polygon([0, 0], avg_radius, irregularity=0.3, spikiness=0.5, num_vertices=10))

    while True:
        x = np.array([np.random.uniform(*xlim), np.random.uniform(*ylim)])
        if pol.exterior_point(x):
            break
    b = pol.boundary_mapping(x)
    n = pol.normal(x)
    tp = pol.tangent_points(x)
    dir = pol.reference_direction(x)

    _, ax = pol.draw()
    ax.plot(*zip(pol.xr(Frame.GLOBAL), x), 'k--o')
    if b is not None:
        ax.plot(*b, 'y+')
        ax.quiver(*b, *n)
    if tp:
        ax.plot(*zip(x, tp[0]), 'g:')
        ax.plot(*zip(x, tp[1]), 'g:')
    ax.quiver(*pol.xr(Frame.GLOBAL), *dir, color='c', zorder=3)

    for i in np.linspace(0, 2 * np.pi, 100):
        x = pol.xr() + 100*np.array([np.cos(i), np.sin(i)])
        b = pol.boundary_mapping(x)
        n = pol.normal(b)
        ax.quiver(*b, *n)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    print("es")

def test_star_primitive_combination():
    n_ellipses = 3
    n_polygons = 2
    n_vertices = 6
    box_width = 2

    ell_axes = [0.7, 0.4]
    pol_box = [box_width, box_width]

    xlim = [-box_width, box_width]
    ylim = [-box_width, box_width]
    ellipses = [Ellipse(ell_axes) for i in range(n_ellipses)]
    polygons = [StarshapedPolygon(generate_convex_polygon(n_vertices, pol_box), is_convex=True)
                for i in range(n_polygons)]
    obstacles = ellipses + polygons
    while True:
        # Generate new positions
        for obs in obstacles:
            obs.set_motion_model(mm.Static(np.random.uniform(-0.2*box_width, 0.2*box_width, 2), np.random.uniform(0, 2*np.pi)))

        # Identify if all obstacle form a single cluster
        kernel = obstacles[0].polygon()
        for obs in obstacles[1:]:
            kernel = kernel.intersection(obs.polygon())

        if not kernel.is_empty:
            break
        else:
            _, ax = plt.subplots()
            [obs.draw(ax=ax, fc='r', alpha=0.2, ec='k') for obs in obstacles]
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            plt.show()

    xr = np.array(kernel.representative_point().coords[0])
    star_obs = StarshapedPrimitiveCombination(obstacles, shapely.geometry.Polygon([]), xr)

    while True:
        x = np.array([np.random.uniform(*xlim), np.random.uniform(*ylim)])
        if star_obs.exterior_point(x):
            break
    b = star_obs.boundary_mapping(x)
    n = star_obs.normal(x)
    tp = star_obs.tangent_points(x)
    dir = star_obs.reference_direction(x)

    _, ax = star_obs.draw()
    draw_shapely_polygon(kernel, ax=ax, fc='g')
    ax.plot(*zip(star_obs.xr(Frame.GLOBAL), x), 'k--o')
    if b is not None:
        ax.plot(*b, 'y+')
        ax.quiver(*b, *n)
    if tp:
        ax.plot(*zip(x, tp[0]), 'g:')
        ax.plot(*zip(x, tp[1]), 'g:')
    ax.quiver(*star_obs.xr(Frame.GLOBAL), *dir, color='c', zorder=3)

    for i in np.linspace(0, 2 * np.pi, 100):
        x = star_obs.xr() + np.array([np.cos(i), np.sin(i)])
        b = star_obs.boundary_mapping(x)
        n = star_obs.normal(b)
        # ax.quiver(*b, *n)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)


if (__name__) == "__main__":
    # test_ellipse()
    # test_nonstar_polygon()
    # test_star_polygon()
    test_star_primitive_combination()
    plt.show()