import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import inspect
import traceback
import shapely
import numpy as np

def tic():
    return time.time()


def toc(t0):
    return (time.time()-t0) * 1000


def draw_shapely_polygon(pol, ax=None, xlim=None, ylim=None, **kwargs):
    if ax is None:
        _, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    pol_list = []
    handles = []
    if pol.geom_type == 'Polygon':
        pol_list += [pol]
    else:
        for p in pol.geoms:
            if p.geom_type == 'Polygon':
                pol_list += [p]
    for p in pol_list:
        if xlim is not None and ylim is not None:
            pol_plot = p.intersection(shapely.geometry.box(xlim[0] - 1, ylim[0] - 1, xlim[1] + 1, ylim[1] + 1))
        else:
            pol_plot = p
        handles += [patches.Polygon(xy=np.vstack((pol_plot.exterior.xy[0], pol_plot.exterior.xy[1])).T, **kwargs)]
        ax.add_patch(handles[-1])
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            ax.set_ylim(ylim)
    return handles, ax


def logprint(message=None, print_stack=0):
  callerframerecord = inspect.stack()[1]    # 0 represents this line
                                            # 1 represents line at caller
  frame = callerframerecord[0]
  info = inspect.getframeinfo(frame)
  if print_stack:
    traceback.print_stack()
  print(info.function + ", line: " + str(info.lineno))
  if message:
      print(message)
