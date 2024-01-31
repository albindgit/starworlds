from obstacles import Frame, StarshapedObstacle, Polygon
import matplotlib.pyplot as plt
import numpy as np
import shapely
from utils import is_cw, is_ccw, is_collinear, orientation_val, get_intersection, is_between, tic, toc


class StarshapedPolygon(Polygon, StarshapedObstacle):

    def __init__(self, polygon, xr=None, **kwargs):
        super().__init__(polygon, xr=xr, **kwargs)
        if xr is None:
            self._compute_kernel()
            if self._kernel.contains(self._kernel.centroid):
                self._xr = np.array(self._kernel.centroid.coords[0])
            else:
                self._xr = np.array(self._kernel.representative_point().coords[0])
        else:
            self._xr = np.array(xr)
        self.vertex_angles = None
        self._update_vertex_angles()
        self.enclosing_ball_diameter = self._polygon.bounds[2]-self._polygon.bounds[0] + self._polygon.bounds[3]-self._polygon.bounds[1]

    def copy(self, id, name):
        if (id == 'duplicate' or id == 'd'):
            id = self.id()
        return StarshapedPolygon(id=id, name=name, polygon=self._polygon, xr=self._xr, motion_model=self._motion_model)

    # Note: Does not recompute the kernel
    def dilated_obstacle(self, padding, id="new", name=None):
        cp = self.copy(id, name)
        cp._polygon = cp._polygon.buffer(padding, cap_style=1, join_style=1)
        cp._pol_bounds = cp._polygon.bounds
        cp.enclosing_ball_diameter = max(cp._polygon.bounds[2]-cp._polygon.bounds[0], cp._polygon.bounds[3]-cp._polygon.bounds[1])
        cp.vertices = np.array(cp._polygon.exterior.coords[:-1])
        cp.circular_vertices = np.array(cp._polygon.exterior.coords)
        cp._polygon_global_pose = None
        cp._polygon_global = None
        cp._update_vertex_angles()
        return cp

    def distance_function(self, x, input_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        dist_center = np.linalg.norm(x_obstacle - self._xr, axis=x.ndim - 1)
        local_radius = np.linalg.norm(self.boundary_mapping(x_obstacle, input_frame=Frame.OBSTACLE, output_frame=Frame.OBSTACLE)
                           - self._xr, axis=x.ndim - 1)
        if dist_center < local_radius:
            # Return proportional inside to have -> [0, 1]
            return (dist_center / local_radius) ** 2
        else:
            return ((dist_center - local_radius) + 1) ** 2


    def boundary_mapping(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        intersect_obstacle = self.line_intersection([self._xr, self._xr+1.1*self.enclosing_ball_diameter*self.reference_direction(x_obstacle, Frame.OBSTACLE, Frame.OBSTACLE)],
                                                    input_frame=Frame.OBSTACLE, output_frame=Frame.OBSTACLE)
        if not intersect_obstacle:
            return None
        if len(intersect_obstacle) == 1:
            return self.transform(intersect_obstacle[0], Frame.OBSTACLE, output_frame)
        else:
            if np.linalg.norm(intersect_obstacle[0]-x_obstacle) < np.linalg.norm(intersect_obstacle[1]-x_obstacle):
                return self.transform(intersect_obstacle[0], Frame.OBSTACLE, output_frame)
            else:
                return self.transform(intersect_obstacle[1], Frame.OBSTACLE, output_frame)

    def normal(self, x, input_frame=Frame.GLOBAL, output_frame=Frame.GLOBAL, x_is_boundary=False, type='edge'):
        x_obstacle = self.transform(x, input_frame, Frame.OBSTACLE)
        angle = np.arctan2(x_obstacle[1]-self._xr[1], x_obstacle[0]-self._xr[0])
        v_idx = np.argmax(self.vertex_angles > angle)

        if v_idx == self.vertices.shape[0]:
            v_idx = 0

        if type == 'edge':
            n_obstacle = np.array([self.vertices[v_idx, 1] - self.vertices[v_idx - 1, 1], self.vertices[v_idx - 1, 0] - self.vertices[v_idx, 0]])
            n_obstacle /= np.linalg.norm(n_obstacle)
        elif type == 'weighted_edges':
            edge_neighbors = [(self.vertices[(v_idx - 2 + i) % self.vertices.shape[0]], self.vertices[(v_idx - 1 + i) % self.vertices.shape[0]]) for i in range(3)]
            edge_neighbors_normal = np.array([[e[1][1] - e[0][1],
                                              e[0][0] - e[1][0]] for e in edge_neighbors])
            edge_closest = [shapely.ops.nearest_points(shapely.geometry.LineString(e),
                                                       shapely.geometry.Point(x_obstacle))[0].coords[0] for e in edge_neighbors]

            dist = [np.linalg.norm(np.array(e)-x_obstacle) for e in edge_closest]
            w = np.array([1/(d+1e-6) for d in dist])
            w /= sum(w)
            n_obstacle = edge_neighbors_normal.T.dot(w)
        return self.rotate(n_obstacle, Frame.OBSTACLE, output_frame)


        #directional weighted mean (see Appendix A) of normal vectors of the surface tiles ni(ξ), the weights wi,
        # and with respect to the reference direction r
        # n_vert = self.vertices.shape[0]
        # r = self.reference_direction(x, input_frame=Frame.OBSTACLE, output_frame=Frame.OBSTACLE)
        # B = np.array((r, (-r[1], r[0])))
        # ws = np.zeros(n_vert)
        # kappa = np.zeros(n_vert)
        # p = 3
        # vi_min = np.inf
        #
        # edge_r = np.zeros((n_vert, 2))
        # _, ax = self.draw(frame=Frame.OBSTACLE)
        # ax.plot(*x_obstacle, 'ko')
        # for i in range(n_vert):
        #     # surface_i = np.array([self.vertices[v_idcs[i] - 1, :], self.vertices[v_idcs[i], :]])
        #     # edge_normal_i = np.array([-(surface_i[0, 1] - surface_i[1, 1]), surface_i[0, 0] - surface_i[1, 0]])
        #     edge_normal_i = np.array([self.vertices[i, 1] - self.vertices[i - 1, 1], self.vertices[i - 1, 0] - self.vertices[i, 0]])
        #     edge_normal_i /= np.linalg.norm(edge_normal_i)
        #     # rotated_edge_normal_i = Rpn.T.dot(edge_normal_i)
        #
        #     pi = np.array(
        #         shapely.ops.nearest_points(shapely.geometry.LineString([self.vertices[i-1, :], self.vertices[i, :]]),
        #                                    shapely.geometry.Point(x_obstacle))[0].coords[0])
        #     # ax.quiver(*pi, *edge_normals[i, :], color='k')
        #     edge_r[i, :] = [np.mean([self.vertices[i, 0], self.vertices[i - 1, 0]]),
        #                     np.mean([self.vertices[i, 1], self.vertices[i - 1, 1]])]
        #     vi = x_obstacle - pi
        #     ei = (vi - edge_normal_i.dot(vi)*edge_normal_i) * np.sign(vi.dot(x_obstacle-edge_r[i, :]))
        #     phi = np.arccos(ei.dot(vi) / (np.linalg.norm(ei) * np.linalg.norm(vi))) * np.sign(edge_normal_i.dot(vi))
        #
        #     if phi > 0:
        #         ws[i] = 0
        #     else:
        #         ws[i] = (np.pi / phi) ** p - 1
        #
        #     # if np.linalg.norm(vi) < vi_min:
        #     #     ws = np.zeros(n_vert)
        #     #     ws[i] = 1
        #     #     vi_min = np.linalg.norm(vi)
        #
        #
        #
        #     # if np.all(np.isclose(pi, x_obstacle)):
        #     #     ws[i] = -1
        #     # else:
        #     #     ws[i] = 1 / np.linalg.norm(x_obstacle - pi) ** 2
        #     rotated_edge_normal_i = B.T.dot(edge_normal_i)
        #
        #     kappa[i] = 0 if rotated_edge_normal_i[0] == 1 else np.arccos(rotated_edge_normal_i[0]) * np.sign(
        #         rotated_edge_normal_i[1])
        #
        #     # v_idx += 1
        # # if np.any(ws < 0):
        # #     ws[np.nonzero(ws > 0)] = 0
        # #     ws[np.nonzero(ws)] = 1
        # #
        # ws = ws / np.sum(ws)
        # # n_obstacle2 =
        #
        # kappa_bar = ws.dot(kappa)
        # tmp_vec = [np.cos(abs(kappa_bar)), np.sin(abs(kappa_bar)) * np.sign(kappa_bar)]
        # n_obstacle = B.dot(tmp_vec)
        #
        # return self.rotate(n_obstacle, Frame.OBSTACLE, output_frame)

    def set_xr(self, xr, input_frame=Frame.OBSTACLE, safe_set=False):
        super().set_xr(xr, input_frame, safe_set)
        self._update_vertex_angles()
    
    def init_plot(self, ax=None, show_reference=True, show_name=False, **kwargs):
        line_handles, ax = super().init_plot(ax=ax, show_name=show_name, **kwargs)
        if ax is None:
            _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
        # Reference point
        line_handles += ax.plot(0, 0, '+', color='k') if show_reference else [None]
        return line_handles, ax

    def update_plot(self, line_handles, frame=Frame.GLOBAL):
        super().update_plot(line_handles, frame)
        if line_handles[1] is not None:
            line_handles[1].set_position(self.xr(frame))
        if line_handles[2] is not None:
            line_handles[2].set_data(*self.xr(frame))

    def _update_vertex_angles(self):
        self.vertex_angles = np.arctan2(self.vertices[:, 1] - self._xr[1], self.vertices[:, 0] - self._xr[0])
        idcs = np.argsort(self.vertex_angles)
        self.vertex_angles = self.vertex_angles[idcs]
        self.vertices = self.vertices[idcs, :]
        self.circular_vertices = np.vstack((self.vertices, self.vertices[0, :]))
        self.vertex_angles = np.hstack((self.vertex_angles, self.vertex_angles[0] + 2 * np.pi))

    def _compute_kernel(self, verbose=False, debug=False):
        if self.is_convex():
            self._kernel = self._polygon
            return

        # Returns true if vertex v[i, :] is reflex
        def is_reflex(i):
            return is_cw(v[i - 1, :], v[i, :], v[(i + 1) % v.shape[0], :])

        # Show polygon
        def draw(xk, F1_idx, L1_idx, i, xk_bounded):
            v_ext = np.vstack((v, v[0, :]))
            xk_ext = np.vstack((xk, xk[0, :])) if xk_bounded else xk
            plt.plot(v_ext[:, 0], v_ext[:, 1], label='v', marker='.')
            # plt.plot(v_ext[:, 0], v_ext[:, 1], label='v')
            # plt.text(v_ext[0, 0], v_ext[0, 1], 'v0')
            plt.text(xk[0, 0], xk[0, 1], 'xk0')
            axes = plt.gca()
            xlim, ylim = axes.get_xlim(), axes.get_ylim()
            plt.plot(xk_ext[:, 0], xk_ext[:, 1], label='xk')
            plt.plot(xk[F1_idx, 0], xk[F1_idx, 1], marker='o', c='c', label='F1')
            end_point = v[i, :] + INF_VAL * (xk[F1_idx, :] - v[i, :]) / np.linalg.norm(xk[F1_idx, :] - v[i, :])
            plt.plot([v[i, 0], end_point[0]], [v[i, 1], end_point[1]], 'c--')
            plt.plot(xk[L1_idx, 0], xk[L1_idx, 1], marker='o', c='b', label='L1')
            end_point = v[i, :] + INF_VAL * (xk[L1_idx, :] - v[i, :]) / np.linalg.norm(xk[L1_idx, :] - v[i, :])
            plt.plot([v[i, 0], end_point[0]], [v[i, 1], end_point[1]], 'b--')
            plt.plot(v[i, 0], v[i, 1], marker='x', c='r', ms=12, label='v{}'.format(i))
            axes.set_xlim(xlim)
            axes.set_ylim(ylim)

        def point_left_of_line(point, line_head, line_tail):
            return is_ccw(line_head, line_tail, point)

        def point_right_of_line(point, line_head, line_tail):
            return is_cw(line_head, line_tail, point)

        pol = shapely.ops.orient(self._polygon)
        v = np.asarray(pol.exterior.coords)[:-1, :]

        # Remove points in line
        v_idcs = np.arange(v.shape[0])
        for i in range(v.shape[0]):
            if is_collinear(v[v_idcs[i - 2], :], v[v_idcs[i - 1], :], v[v_idcs[i], :]):
                v_idcs[i - 1] = v_idcs[i - 2]
        v_idcs = np.unique(v_idcs)
        v = v[v_idcs, :]
        if is_collinear(v[-2, :], v[-1, :], v[0, :]):
            v = v[:-1]

        N = v.shape[0]
        for i in range(N):
            # Order v to have v[:, 0] as reflex vertex
            if is_reflex(i):
                v = np.vstack((v[i:, :], v[:i, :]))
                break

        INF_VAL = 300.
        # Initial step
        F1 = v[0, :] + INF_VAL * (v[0, :] - v[1, :]) / np.linalg.norm(v[0, :] - v[1, :])
        L1 = v[0, :] + INF_VAL * (v[0, :] - v[-1, :]) / np.linalg.norm(v[0, :] - v[-1, :])
        xk = np.vstack((F1, v[0, :], L1))
        F1_idx = 0  # Index of point F1 in xk
        L1_idx = 2  # Index of point L1 in xk
        xk_bounded = False

        if verbose:
            print("------")
            print(0)
            print("F1:", end=" ")
            print(F1)
            print("L1:", end=" ")
            print(L1)
        if debug:
            draw(xk, F1_idx, L1_idx, 0, xk_bounded)
            plt.show()

        for i in range(1, N):

            if verbose:
                print("------")
                print(i)
                print("F1 index: {}".format(F1_idx))
                print("L1 index: {}".format(L1_idx))

            L1 = xk[L1_idx, :]
            F1 = xk[F1_idx, :]
            vi = v[i, :]
            vi_1 = v[(i + 1) % N, :]

            # (1) vi is reflex
            if is_reflex(i):
                # (1.2) F1 lies to the left of line ->(vi->vi+1)->vi+1
                start_point = vi - INF_VAL * (vi_1 - vi) / np.linalg.norm(vi_1 - vi)
                if point_left_of_line(F1, start_point, vi_1):
                    case = 2
                    if debug:
                        draw(xk, F1_idx, L1_idx, i, xk_bounded)
                        plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*', color='y',
                                 label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                        plt.title("F1 lies to the left of line ->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                        plt.show()
                    if verbose:
                        print("(1.2) - F1 ({},{}) lies to the left of line ->(v{}->v{})->v{}".format(F1[0], F1[0], i,
                                                                                                     i + 1, i + 1))
                # (1.1) F1 lies on or to the right of line ->(vi->vi+1)->vi+1
                else:
                    case = 1
                    if verbose:
                        print(
                            "(1.1) - F1 ({},{}) lies on or to the right of line ->(v{}->v{})->v{}".format(F1[0], F1[0],
                                                                                                          i, i + 1,
                                                                                                          i + 1))
                        print("Scan xk ccw from F1 until we reach edge intersecting line ->(v{}->v{})->v{}".format(i,
                                                                                                                   i + 1,
                                                                                                                   i + 1))
                    # Scan xk ccw from F1 to L1 until we reach edge intersecting line ->(vi->vi+1)->vi+1
                    w1 = None
                    idx_offsets = range(1, xk.shape[0] + 1)
                    for off in idx_offsets:
                        t = (F1_idx + off) % xk.shape[0]
                        if not xk_bounded and t == 0:
                            break
                        # Get intersection w1 of current edge and line ->(vi->vi+1)->vi+1
                        wt_prev = xk[t - 1, :]
                        wt = xk[t, :]
                        w1 = get_intersection(wt_prev, wt, start_point, vi_1)
                        if debug:
                            draw(xk, F1_idx, L1_idx, i, xk_bounded)
                            col = 'r' if w1 is None else 'g'
                            plt.plot([wt_prev[0], wt[0]], [wt_prev[1], wt[1]], marker='*', color=col, label="edge")
                            plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*', color='y',
                                     label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                            plt.title(
                                "Scan xk ccw from F1 until we reach edge intersecting line ->(v{}->v{})->v{}".format(i,
                                                                                                                     i + 1,
                                                                                                                     i + 1))
                        if w1 is not None:
                            if debug:
                                plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                plt.legend()
                                plt.show()
                            break
                        if debug:
                            plt.legend()
                            plt.show()

                        if t == L1_idx:
                            break
                    # If no intersecting line is reached no kernel exists
                    if w1 is None:
                        # if debug:
                        draw(xk, F1_idx, L1_idx, i, xk_bounded)
                        plt.title('No kernel found.. Polygon not starshaped.')
                        plt.legend()
                        plt.show()
                        return False
                    if verbose:
                        print("Found intersection ({},{}) at line ->(v{}->v{})->v{}".format(w1[0], w1[1], i, i + 1,
                                                                                            i + 1))
                        print("Scan xk cw from F1 until we reach edge intersecting line ->(v{}->v{})->v{}".format(i,
                                                                                                                  i + 1,
                                                                                                                  i + 1))
                    # Scan xk cw from F1 until we reach edge intersecting line ->(vi->vi+1)->vi+1
                    w2 = None
                    idcs_list = np.flip(np.roll(np.arange(xk.shape[0]), -F1_idx - 1)) if xk_bounded else range(F1_idx,
                                                                                                               0, -1)
                    # for s in range(F1_idx, 0, -1):
                    for s in idcs_list:
                        ws = xk[s, :]
                        ws_prev = xk[s - 1, :]
                        # Get intersection w2 of edge and line ->(vi->vi+1)->vi+1
                        w2 = get_intersection(ws_prev, ws, start_point, vi_1)

                        if debug:
                            draw(xk, F1_idx, L1_idx, i, xk_bounded)
                            plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                            col = 'r' if w2 is None else 'g'
                            plt.plot([ws_prev[0], ws[0]], [ws_prev[1], ws[1]], marker='*', color=col, label="edge")
                            plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*', color='y',
                                     label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                            plt.title(
                                "Scan xk cw from F1 until we reach edge intersecting line ->(v{}->v{})->v{}".format(i,
                                                                                                                    i + 1,
                                                                                                                    i + 1))

                        if w2 is not None:
                            if xk_bounded and s > t:
                                alpha = xk[xk.shape[0]:, :]  # Empty array
                                beta = xk[t:s, :]
                                L1_idx = L1_idx - t + 2
                            else:
                                alpha = xk[:s, :]
                                beta = xk[t:, :]
                                L1_idx = L1_idx - t + s + 2
                            if debug:
                                print("xk: ", end=" ")
                                print(xk)
                                print("alpha: ", end=" ")
                                print(alpha)
                                print("beta: ", end=" ")
                                print(beta)
                                print("w1: ", end=" ")
                                print(w1)
                                print("w2: ", end=" ")
                                print(w2)
                                plt.plot(w1[0], w1[1], 's', label="w1", color='k')
                                plt.plot(w2[0], w2[1], 's', label="w2", color='g')
                                plt.plot(alpha[:, 0], alpha[:, 1], 'r--', label="alpha")
                                plt.plot(beta[:, 0], beta[:, 1], 'k--', label="beta")
                                plt.legend()
                                plt.show()
                            xk = np.vstack((alpha, w2, w1, beta))
                            w1_idx = alpha.shape[0] + 1
                            w2_idx = alpha.shape[0]
                            ####### F1 index reassignment should not be necessary for this case
                            F1_idx = w2_idx
                            # L1_idx = L1_idx - t + s + 2

                            if verbose:
                                print("Update 1")
                                print("Found intersection ({},{}) at line ->(v{}->v{})->v{}".format(w2[0], w2[1], i,
                                                                                                    i + 1, i + 1))
                            break
                        if debug:
                            plt.legend()
                            plt.show()
                    # If no intersecting line is reached
                    if w2 is None:
                        # Test if xk+1 is bounded
                        # If slope ->(vi->vi+1)->vi+1 is comprised between the slopes of initial and final half lines of xk,
                        if (orientation_val(xk[-2, :], xk[-1, :], start_point) * orientation_val(vi_1, start_point,
                                                                                                 xk[0, :])) > 0:
                            beta = xk[t:, :]
                            if debug:
                                draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                plt.plot(xk[:2, 0], xk[:2, 1], '--c', label="initial half line")
                                plt.plot(xk[-2:, 0], xk[-2:, 1], '--m', label="final half line")
                                plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*', color='y',
                                         label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                                plt.title(
                                    "->(v{}->v{})->v{} between initial and finial half lines of xk".format(
                                        i, i + 1, i + 1))
                                plt.plot(beta[:, 0], beta[:, 1], 'k--', label="beta")
                                plt.legend()
                                plt.show()
                            if verbose:
                                print("Update 2 - xk still unbounded")
                            # then xk+1= ->(vi->vi+1)->w1->beta is also unbounded.
                            xk = np.vstack((start_point, w1, beta))
                            w1_idx = 1
                            # xk_bounded = False
                            F1_idx = 0
                            L1_idx -= t - 1


                        else:
                            # otherwise scan xk cw from xk[-1,:] until we reach edge intersecting line ->(vi->vi+1)->vi+1
                            if verbose:
                                print(
                                    "Scan xk cw from end until we reach edge intersecting line ->(v{}->v{})->v{}".format(
                                        i, i + 1, i + 1))
                            w2 = None
                            if xk_bounded:
                                r = xk.shape[0]
                                w2 = get_intersection(xk[0, :], xk[r - 1, :], start_point, vi_1)
                                if debug:
                                    draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                    plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                    col = 'r' if w2 is None else 'g'
                                    plt.plot([xk[r - 1, 0], xk[0, 0]], [xk[r - 1, 1], xk[0, 1]], marker='*', color=col,
                                             label="edge")
                                    plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*',
                                             color='y',
                                             label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                                    plt.title(
                                        "Scan xk cw from xk[-1, :] until we reach edge intersecting line ->(v{}->v{})->v{}".format(
                                            i, i + 1, i + 1))
                                    if w2 is not None:
                                        delta = xk[t:r, :]
                                        if debug:
                                            print(t, r)
                                            plt.plot(delta[:, 0], delta[:, 1], 'k--', label="delta")
                                            plt.plot(w1[0], w1[1], 's', label="w1", color='k')
                                            plt.plot(w2[0], w2[1], 's', label="w2", color='g')
                                            plt.legend()
                                            plt.show()
                                    if debug:
                                        plt.legend()
                                        plt.show()
                            if w2 is None:
                                for r in range(xk.shape[0] - 1, 0, -1):
                                    # Get intersection w2 of edge and line ->(vi->vi+1)->vi+1
                                    w2 = get_intersection(xk[r, :], xk[r - 1, :], start_point, vi_1)
                                    if debug:
                                        draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                        plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                        col = 'r' if w2 is None else 'g'
                                        plt.plot([xk[r - 1, 0], xk[r, 0]], [xk[r - 1, 1], xk[r, 1]], marker='*',
                                                 color=col, label="edge")
                                        plt.plot([start_point[0], vi_1[0]], [start_point[1], vi_1[1]], marker='*',
                                                 color='y', label="->(v{}->v{})->v{}".format(i, i + 1, i + 1))
                                        plt.title(
                                            "Scan xk cw from xk[-1, :] until we reach edge intersecting line ->(v{}->v{})->v{}".format(
                                                i, i + 1, i + 1))
                                    if w2 is not None:
                                        delta = xk[t:r, :]
                                        if debug:
                                            print(t, r)
                                            plt.plot(delta[:, 0], delta[:, 1], 'k--', label="delta")
                                            plt.plot(w1[0], w1[1], 's', label="w1", color='k')
                                            plt.plot(w2[0], w2[1], 's', label="w2", color='g')
                                            plt.legend()
                                            plt.show()
                                        break
                                    if debug:
                                        plt.legend()
                                        plt.show()
                            if verbose:
                                print("Update 3")
                                print("Found intersection ({},{}) at line ->(v{}->v{})->v{}".format(w2[0], w2[1], i,
                                                                                                    i + 1, i + 1))
                            # Set xk as delta-w2-w1
                            xk = np.vstack((delta, w2, w1))
                            w1_idx = delta.shape[0] + 1
                            w2_idx = delta.shape[0]
                            xk_bounded = True

                            F1_idx = 0
                            L1_idx = min(L1_idx - t, xk.shape[0] - 1)

                # F1 update
                if case == 1:
                    # If ->(vi->vi+1)->vi+1 has just one intersection with xk F1 = startpoint
                    if w2 is None:
                        F1_idx = 0
                    # Otherwise F1 = w2
                    else:
                        F1_idx = w2_idx
                if case == 2:
                    # Scan xk ccw from F1 until find vertex wt s.t. wt+1 lies to the
                    # right of vi+1->(vi+1->wt)->. Let F1 = wt.
                    idx_offsets = range(xk.shape[0])
                    for off in idx_offsets:
                        t = (F1_idx + off) % xk.shape[0]
                        w_next = xk[(t + 1) % xk.shape[0], :]
                        line_end_point = vi_1 + INF_VAL * (xk[t, :] - vi_1)
                        if point_right_of_line(w_next, vi_1, line_end_point):
                            F1_idx = t
                            break

                # Check update of previous L1 index for new xk
                # if not np.isclose(np.linalg.norm(L1 - xk[L1_idx, :]), 0):
                # print("BAD L1 update [{},{}] -> [{},{}]".format(xk[L1_idx, 0], xk[L1_idx, 1], L1[0], L1[1]))
                # plt.figure(2)
                # draw(xk, F1_idx, L1_idx, i, xk_bounded)
                # plt.show()

                # L1 update
                # scan xk ccw from L1 until find vertex wu s.t. wu+1 lies to the
                # left of vi+1->(vi+1->wu)->. Let L1 = wu.
                idx_offsets = range(xk.shape[0])
                for off in idx_offsets:
                    u = (L1_idx + off) % xk.shape[0]
                    w_next = xk[(u + 1) % xk.shape[0], :]
                    line_end_point = vi_1 + INF_VAL * (xk[u, :] - vi_1)
                    if point_left_of_line(w_next, vi_1, line_end_point):
                        L1_idx = u
                        break

            else:
                if verbose:
                    print("(2)")
                # Endpoint for line vi->(vi->vi+1)->
                end_point = vi + INF_VAL * (vi_1 - vi) / np.linalg.norm(vi_1 - vi)

                # (2.2) L1 lies to the left of line vi->(vi->vi+1)->
                if point_left_of_line(L1, vi, end_point):
                    case = 2
                    if verbose:
                        print("(2.2) - L1 ({},{}) lies to the left of line v{}->(v{}->v{})->".format(L1[0], L1[1], i, i,
                                                                                                     i + 1))
                    # xk stays the same

                    if debug:
                        draw(xk, F1_idx, L1_idx, i, xk_bounded)
                        plt.plot([end_point[0], vi[0]], [end_point[1], vi[1]], marker='*', color='y',
                                 label="v{}->(v{}->v{})->".format(i, i, i + 1))
                        plt.title("L1 lies to the left of line v{}->(v{}->v{})->".format(i, i, i + 1))
                        plt.show()

                # (2.1) L1 lies on or to the right of line vi->(vi->vi+1)->
                else:
                    case = 1
                    if verbose:
                        print(
                            "(2.1) - L1 ({},{}) lies on or to the right of line v{}->(v{}->v{})->".format(L1[0], L1[1],
                                                                                                          i, i, i + 1))
                        print(
                            "Scan xk cw from L1 until we reach F1 or an edge intersecting line v{}->(v{}->v{})->".format(
                                i, i, i + 1))
                    # Scan xk cw from L1 until we reach F1 or an edge intersecting line vi->(vi->vi+1)->
                    idx_offsets = range(xk.shape[0])
                    w1 = None
                    for off in idx_offsets:
                        t = L1_idx - off
                        # If circular
                        if t < 0:
                            t += xk.shape[0]
                        if t == F1_idx:
                            break
                        # Get intersection w1 of edge and line vi->(vi->vi+1)->
                        w1 = get_intersection(xk[t, :], xk[t - 1, :], vi, end_point)
                        if debug:
                            draw(xk, F1_idx, L1_idx, i, xk_bounded)
                            col = 'r' if w1 is None else 'g'
                            plt.plot([xk[t - 1, 0], xk[t, 0]], [xk[t - 1, 1], xk[t, 1]], marker='*', color=col,
                                     label="edge")
                            plt.plot([end_point[0], v[i, 0]], [end_point[1], v[i, 1]], marker='*', color='y',
                                     label="v{}->(v{}->v{})->".format(i, i, i + 1))
                            plt.title(
                                "Scan xk cw from L1 until we reach F1 or an edge intersecting line v{}->(v{}->v{})->".format(
                                    i,
                                    i,
                                    i + 1))
                        if w1 is not None:
                            if debug:
                                plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                plt.legend()
                                plt.show()
                            break
                        if debug:
                            plt.legend()
                            plt.show()
                    # If no intersecting line is reached no kernel exists
                    if w1 is None:
                        # if debug:
                        draw(xk, F1_idx, L1_idx, i, xk_bounded)
                        plt.title('No kernel found. Polygon not starshaped.')
                        plt.show()
                        return False

                    if verbose:
                        print("Found intersection ({},{}) at line v{}->(v{}->v{})-> for edge ({},{})-({},{})".format(
                            w1[0], w1[1], i, i, i + 1, xk[t - 1, 0], xk[t - 1, 1], xk[t, 0], xk[t, 1]))
                        print("Scan xk ccw from L1 until we reach edge intersecting line v{}->(v{}->v{})->".format(i,
                                                                                                                   i,
                                                                                                                   i + 1))
                    # Scan xk ccw from L1 until we reach edge w2 intersecting line vi->(vi->vi+1)->
                    w2 = None
                    idx_offsets = range(1, xk.shape[0] + 1)
                    for off in idx_offsets:
                        s = (L1_idx + off) % xk.shape[0]
                        if not xk_bounded and s == 0:
                            break
                        # Get intersection w2 of edge and line vi->(vi->vi+1)->
                        w2 = get_intersection(xk[s - 1, :], xk[s, :], vi, end_point)
                        if debug:
                            draw(xk, F1_idx, L1_idx, i, xk_bounded)
                            col = 'r' if w2 is None else 'g'
                            plt.plot([xk[s - 1, 0], xk[s, 0]], [xk[s - 1, 1], xk[s, 1]], marker='*', color=col,
                                     label="edge")
                            plt.plot([end_point[0], v[i, 0]], [end_point[1], v[i, 1]], marker='*', color='y',
                                     label="v{}->(v{}->v{})->".format(i, i, i + 1))
                            plt.title(
                                "Scan xk ccw from L1 until we reach edge intersecting line v{}->(v{}->v{})->".format(i,
                                                                                                                     i,
                                                                                                                     i + 1))
                        if w2 is not None:
                            if xk_bounded and t > s:
                                alpha = xk[xk.shape[0]:, :]  # Empty array
                                beta = xk[s:t, :]
                            else:
                                alpha = xk[:t, :]
                                beta = xk[s:, :]
                            if debug:
                                print("xk: ", end=" ")
                                print(xk)
                                print("alpha: ", end=" ")
                                print(alpha)
                                print("beta: ", end=" ")
                                print(beta)
                                print("w1: ", end=" ")
                                print(w1)
                                print("w2: ", end=" ")
                                print(w2)
                                plt.plot(w1[0], w1[1], 's', label="w1", color='k')
                                plt.plot(w2[0], w2[1], 's', label="w2", color='g')
                                plt.plot(alpha[:, 0], alpha[:, 1], 'r--', label="alpha")
                                plt.plot(beta[:, 0], beta[:, 1], 'k--', label="beta")
                                plt.legend()
                                plt.show()
                            if verbose:
                                print("Update 1")
                                print(
                                    "Found intersection ({},{}) at line v{}->(v{}->v{})-> for edge ({},{})-({},{})".format(
                                        w2[0], w2[1], i, i, i + 1, xk[s, 0], xk[s, 1], xk[(s + 1) % xk.shape[0], 0],
                                        xk[(s + 1) % xk.shape[0], 1]))

                            xk = np.vstack((alpha, w1, w2, beta))
                            w1_idx = alpha.shape[0]
                            w2_idx = alpha.shape[0] + 1
                            break
                        if debug:
                            plt.legend()
                            plt.show()
                    # If no intersecting line is reached
                    if w2 is None:
                        # Test if xk+1 is bounded
                        # If slope vi->(vi->vi+1)-> is comprised between the slopes of initial and final half lines of xk,
                        if not xk_bounded and ((orientation_val(xk[-2, :], xk[-1, :], end_point) * orientation_val(vi,
                                                                                                                   end_point,
                                                                                                                   xk[0,
                                                                                                                   :])) > 0):
                            alpha = xk[:t, :]
                            if debug:
                                draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                plt.plot(w1[0], w1[1], 's', label="w1", color='g')
                                plt.plot(xk[:2, 0], xk[:2, 1], '--c', label="initial half line")
                                plt.plot(xk[-2:, 0], xk[-2:, 1], '--m', label="final half line")
                                plt.plot([end_point[0], v[i, 0]], [end_point[1], v[i, 1]], marker='*', color='y',
                                         label="v{}->(v{}->v{})->".format(i, i, i + 1))
                                plt.title(
                                    "v{}->(v{}->v{})-> between initial and final half lines of xk".format(
                                        i, i, i + 1))
                                plt.plot(alpha[:, 0], alpha[:, 1], 'r--', label="alpha")
                                plt.legend()
                                plt.show()
                            if verbose:
                                print("Update 2 - xk still unbounded")
                            # then xk+1= alpha->w1->(vi->vi+1)-> is also unbounded.
                            xk = np.vstack((alpha, w1, end_point))
                            w1_idx = alpha.shape[0]
                            L1_idx = xk.shape[0] - 1
                            F1_idx = 0
                        else:
                            if verbose:
                                print(
                                    "Scan xk ccw from start until we reach edge intersecting line v{}->(v{}->v{})->".format(
                                        i,
                                        i,
                                        i + 1))
                            # otherwise scan xk ccw from xk[0,:] until we reach edge intersecting line vi->(vi->vi+1)->
                            w2 = None
                            for r in range(1, xk.shape[0] + 1):
                                # Get intersection w2 of edge and line vi->(vi->vi+1)->
                                w2 = get_intersection(xk[r - 1, :], xk[r % xk.shape[0], :], vi, end_point)
                                if debug:
                                    draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                    col = 'r' if w2 is None else 'g'
                                    plt.plot([xk[r - 1, 0], xk[r % xk.shape[0], 0]],
                                             [xk[r - 1, 1], xk[r % xk.shape[0], 1]], marker='*', color=col,
                                             label="edge")
                                    plt.plot([end_point[0], v[i, 0]], [end_point[1], v[i, 1]], marker='*', color='y',
                                             label="v{}->(v{}->v{})->".format(i, i, i + 1))
                                    plt.title(
                                        "Scan xk ccw from xk[0,:] until we reach edge intersecting line v{}->(v{}->v{})->".format(
                                            i,
                                            i,
                                            i + 1))
                                if w2 is not None:
                                    delta = xk[r:t, :]
                                    if debug:
                                        plt.plot(w2[0], w2[1], 's', label="w2", color='g')
                                        plt.legend()
                                        plt.show()
                                    break
                                if debug:
                                    plt.legend()
                                    plt.show()
                            if verbose:
                                print("Update 3")
                                print("Found intersection ({},{}) at line v{}->(v{}->v{})->".format(w2[0], w2[1], i, i,
                                                                                                    i + 1))
                            # Set xk as delta-w1-vi-vi+1-w2
                            xk = np.vstack((delta, w1, w2))
                            w1_idx = delta.shape[0]
                            w2_idx = delta.shape[0] + 1
                            xk_bounded = True
                            F1_idx = 0
                            L1_idx = min(L1_idx - t, xk.shape[0] - 1)

                # F1 update

                # If vi+1 in vi->(vi->vi+1)->w1,
                if case == 2 or is_between(vi, vi_1, w1):
                    # scan xk ccw from F1 until find vertex wt s.t. wt+1 lies to the
                    # right of vi+1->(vi+1->wt)->. Let F1 = wt.
                    idx_offsets = range(xk.shape[0])
                    for off in idx_offsets:
                        t = (F1_idx + off) % xk.shape[0]
                        w_next = xk[(t + 1) % xk.shape[0], :]
                        line_end_point = vi_1 + INF_VAL * (xk[t, :] - vi_1)
                        if debug:
                            draw(xk, F1_idx, L1_idx, i, xk_bounded)
                            plt.plot([vi_1[0], line_end_point[0]], [vi_1[1], line_end_point[1]], '--',
                                     label='v{}->(v{}->wt)'.format(i + 1, i + 1))
                            c = 'g' if point_right_of_line(w_next, vi_1, line_end_point) else 'r'
                            plt.plot(w_next[0], w_next[1], color=c, marker='s', label='w_t+1')
                            plt.title(
                                "Update F1\n scan xk ccw from F1 until find vertex wt s.t. wt+1 lies to the right of v{}->(v{}->wt)->.".format(
                                    i + 1, i + 1))
                            plt.legend()
                            plt.show()
                        if point_right_of_line(w_next, vi_1, line_end_point):
                            F1_idx = t
                            break
                else:
                    F1_idx = w1_idx

                # L1 update
                if case == 1:
                    if w2 is not None:
                        # If vi+1 in vi->(vi->vi+1)->w2
                        if is_between(vi, vi_1, w2):
                            L1_idx = w2_idx
                        else:
                            # scan xk ccw from w2 until find vertex wu s.t. wu+1 lies to the
                            # left of vi+1->(vi+1->wu)->. Let L1 = wu.
                            idx_offsets = range(xk.shape[0])
                            for off in idx_offsets:
                                u = (w2_idx + off) % xk.shape[0]
                                w_next = xk[(u + 1) % xk.shape[0], :]
                                line_end_point = vi_1 + INF_VAL * (xk[u, :] - vi_1)
                                if debug:
                                    draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                    plt.plot([vi_1[0], line_end_point[0]], [vi_1[1], line_end_point[1]], '--',
                                             label='v{}->(v{}->wt)'.format(i + 1, i + 1))
                                    c = 'g' if point_left_of_line(w_next, vi_1, line_end_point) else 'r'
                                    plt.plot(w_next[0], w_next[1], color=c, marker='s', label='w_t+1')
                                    plt.title(
                                        "Update L1\n scan xk ccw from w2 until find vertex wu s.t. wu+1 lies to the left of v{}->(v{}->wt)->.".format(
                                            i + 1, i + 1))
                                    plt.legend()
                                    plt.show()
                                if point_left_of_line(w_next, vi_1, line_end_point):
                                    L1_idx = u
                                    break

                    # (2.1.2) line vi->(vi->vi+1)-> intersects xk just in w1
                    else:
                        L1_idx = xk.shape[0] - 1
                if case == 2:
                    # print("(2.1.2)")
                    if xk_bounded:
                        # scan xk ccw from w2 until find vertex wu s.t. wu+1 lies to the
                        # left of vi+1->(vi+1->wu)->. Let L1 = wu.
                        idcs_list = np.roll(np.arange(xk.shape[0]), -L1_idx) if xk_bounded else range(L1_idx,
                                                                                                      xk.shape[0] - 1)
                        for u in idcs_list:
                            w_next = xk[(u + 1) % xk.shape[0], :]
                            line_end_point = vi_1 + INF_VAL * (xk[u, :] - vi_1)
                            if debug:
                                draw(xk, F1_idx, L1_idx, i, xk_bounded)
                                plt.plot([vi_1[0], line_end_point[0]], [vi_1[1], line_end_point[1]], '--',
                                         label='v{}->(v{}->wt)'.format(i + 1, i + 1))
                                c = 'g' if point_left_of_line(w_next, vi_1, line_end_point) else 'r'
                                plt.plot(w_next[0], w_next[1], color=c, marker='s', label='w_t+1')
                                plt.title(
                                    "Update L1\n scan xk ccw from w2 until find vertex wu s.t. wu+1 lies to the left of v{}->(v{}->wt)->.".format(
                                        i + 1, i + 1))
                                plt.legend()
                                plt.show()
                            if point_left_of_line(w_next, vi_1, line_end_point):
                                L1_idx = u
                                break

            if verbose:
                print("F1:", end=" ")
                print(F1)
                print("L1:", end=" ")
                print(L1)
                print("xk:", end=" ")
                print(xk)
        if debug or verbose:
            draw(xk, F1_idx, L1_idx, 0, xk_bounded)
            plt.show()
        _, idx = np.unique(xk, axis=0, return_index=True)
        xk = xk[np.sort(idx), :]
        self._kernel = shapely.geometry.Polygon(xk)


