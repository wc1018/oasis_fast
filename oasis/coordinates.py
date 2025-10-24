import numpy as np

from oasis.common import get_np_unit_dytpe


def cartesian_product(arrays: list[np.ndarray]) -> np.ndarray:
    """Generalized N-dimensional products
    Taken from https://stackoverflow.com/questions/11144513/
    Answer by Nico Schlömer
    Updated for numpy > 1.25

    Parameters
    ----------
    arrays : list[numpy.ndarray]
        Arrays for cartesian product `arr1 X arr2 X ... arrN`, where N is the 
        number of arrays and the dimension of the product.

    Returns
    -------
    numpy.ndarray
        N-dimensional cartesian product.
    """
    len_arr = len(arrays)
    dtype = np.result_type(*[a.dtype for a in arrays])
    arr = np.empty([len(a) for a in arrays] + [len_arr], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[..., i] = a
    return arr.reshape(-1, len_arr)


def gen_data_pos_regular(boxsize: float, gridsize: float) -> np.ndarray:
    """Populate coordinates with one particle per subbox at the centre in steps
    of nside between particles.

    Parameters
    ----------
    boxsize : float
        Length of the box.
    gridsize : float
        Length of the grid.

    Returns
    -------
    numpy.ndarray

    """
    # Number of cells per side.
    n_per_side = np.int_(np.ceil(boxsize / gridsize))

    # Determine data type for integer arrays based on the maximum number of
    # elements.
    uint_dtype = get_np_unit_dytpe(n_per_side)

    # Set of natural numbers from 0 to N-1.
    n_range = np.arange(n_per_side, dtype=uint_dtype)

    # Set of index vectors. Each vector points to the (i, j, k)-th cell.
    pos = np.int_(cartesian_product([n_range, n_range, n_range]))
    centre = gridsize * (pos + 0.5)
    return centre


def gen_data_pos_random(
    boxsize: float,
    nsamples: int,
    seed: int = None
) -> np.ndarray:
    """Generate random data points inside a cubic box.

    Parameters
    ----------
    boxsize : float
        Length of the box.
    nsamples : int
        Number of points to sample.
    seed : int, optional
        Random gnerator seed, by default None.

    Returns
    -------
    numpy.ndarray

    """
    np.random.seed(seed=seed)
    return boxsize * np.random.uniform(0, 1, (nsamples, 3))


def relative_coordinates(
    x: np.ndarray,
    x0: np.ndarray,
    boxsize: float,
    periodic: bool = True
) -> float:
    """Returns the coordinates x relative to x0 accounting for periodic boundary
    conditions.

    Parameters
    ----------
    x : np.ndarray
        Position array (N, 3).
    x0 : np.ndarray
        Reference position in cartesian coordinates.
    boxsize : float
        Size of simulation box.
    periodic : bool, optional
        Set to True if the simulation box is periodic, by default True.

    Returns
    -------
    float
        Relative positions.
    """
    if type(x) in [list, tuple]:
        raise TypeError("Input 'x' must be an array (not a list or tuple)")
    if periodic:
        return (x - x0 + 0.5 * boxsize) % boxsize - 0.5 * boxsize
    return x - x0


def velocity_components(
    pos: np.ndarray,
    vel: np.ndarray,
) -> tuple[np.ndarray]:
    """Computes the radial and tangential velocites from cartesian/rectangular 
    coordinates.

    Parameters
    ----------
    pos : np.ndarray
        Cartesian coordinates.
    vel : np.ndarray
        Cartesian velocities.

    Returns
    -------
    tuple[np.ndarray]
        Radial velocity, tangential velocity and magnitude squared of the 
        velocity.
    """
    if np.ndim(pos) < 2:
        raise ValueError(f'Number of dimensions is {np.ndim(pos)}. Please ' +
                         'reshape array to match 2D.')

    # Transform coordinates from cartesian to spherical
    #   rs = sqrt( x**2 + y**2 + z**2 )
    rps = np.sqrt(np.sum(np.square(pos), axis=1))
    #   theta = arccos( z / rs )
    thetas = np.arccos(pos[:, 2] / rps)
    #   phi = arctan( y / x)
    phis = np.arctan2(pos[:, 1], pos[:, 0])

    # Get radial vector in cartesian coordinates
    rp_hat = np.zeros_like(pos)
    rp_hat[:, 0] = np.sin(thetas) * np.cos(phis)
    rp_hat[:, 1] = np.sin(thetas) * np.sin(phis)
    rp_hat[:, 2] = np.cos(thetas)

    # Get tangential vector in cartesian coordinates
    rt_hat = np.zeros_like(pos)
    rt_hat[:, 0] = np.cos(thetas) * np.cos(phis)
    rt_hat[:, 1] = np.cos(thetas) * np.sin(phis)
    rt_hat[:, 2] = -np.sin(thetas)

    # Compute radial velocity as v dot r_hat
    vr = np.sum(vel * rp_hat, axis=1)
    vt = np.sum(vel * rt_hat, axis=1)
    # Velocity squared
    v2 = np.sum(np.square(vel), axis=1)

    return vr, vt, v2


###
