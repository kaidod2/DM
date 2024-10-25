import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, ticker
from matplotlib.patches import Ellipse
from matplotlib import transforms
import matplotlib.cm

VIRIDIS_REV = cm.viridis.reversed()


def normalize(x: np.ndarray,
              ord: int | float | str = 2,
              axis: int | tuple[int, int] = None,
              keepdims: bool = False) -> np.ndarray:
    nmlzd_x = np.divide(x, np.linalg.norm(x, ord, axis, keepdims))
    nmlzd_x = np.where(np.abs(nmlzd_x) < 1e-16, 0, nmlzd_x)
    return nmlzd_x


def plot_surface(fig, f, xlim, ylim, zlim, xstride, ystride, subplot_coords=None):
    X, Y, Z = make_3d_data(f, xlim, ylim, xstride, ystride)
    if subplot_coords is not None:
