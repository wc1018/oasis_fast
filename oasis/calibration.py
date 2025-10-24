import os
from functools import partial
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

import h5py as h5
import numpy as np
from scipy.optimize import curve_fit, minimize
from tqdm import tqdm

from oasis.common import G_gravity
from oasis.coordinates import relative_coordinates, velocity_components
from oasis.minibox import get_mini_box_id, load_particles, load_objects

from scipy.spatial import cKDTree


def _get_candidate_particle_data(
    mini_box_id: int,
    pos_seed: list[np.ndarray],
    vel_seed: list[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    padding: float = 5.0,
) -> np.ndarray:
    """
    Extracts all requested seeds from a single minibox.

    Parameters
    ----------
    mini_box_id : int
        Mini-box ID
    pos_seed : list[np.ndarray]
        Seed positions
    vel_seed : list[np.ndarray]
        Seed velocities
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe

    Returns
    -------
    np.ndarray
        Particle's radial distance, radial velocity, and log of the square
        of the velocity in units of R200m and M200m.
    """

    # Load particles in the minibox
    pos, vel, *_ = load_particles(mini_box_id, boxsize, minisize, save_path, padding)

    # Iterate over seeds in current mini box
    r, vr, lnv2 = ([] for _ in range(3))

    for i in range(len(pos_seed)):
        # Compute the relative positions of all particles in the box
        rel_pos = relative_coordinates(pos, pos_seed[i], boxsize)

        # Only work with those close to the seed
        mask_close = np.prod(np.abs(rel_pos) <= r_max, axis=1, dtype=bool)
        rel_pos = rel_pos[mask_close]
        rel_vel = vel[mask_close] - vel_seed[i]

        # Compute radial distance, radial and tangential velocities
        rps = np.sqrt(np.sum(np.square(rel_pos), axis=1))
        vrp, _, v2p = velocity_components(rel_pos, rel_vel)

        # Compute R200m and M200m
        rps_prof = rps[np.argsort(rps)]
        mass_prof = part_mass * np.arange(1, len(rps_prof) + 1)

        # Average density profile
        rho_prof = mass_prof / ((4.0 / 3.0) * np.pi * rps_prof**3)
        rho_target = 200.0 * rhom

        # Find the first radius where rho <= 200 * rhom
        cross = np.where(rho_prof <= rho_target)[0]

        if len(cross) == 0:
            # If density never drops below 200 * rhom, take the outermost point
            r200m = rps_prof[-1]
            m200m = mass_prof[-1]
        else:
            idx = cross[0]
            if idx == 0:
                r200m = rps_prof[0]
                m200m = mass_prof[0]
            else:
                # Linear interpolation to find r200m
                r1, r2 = rps_prof[idx - 1], rps_prof[idx]
                rho1, rho2 = rho_prof[idx - 1], rho_prof[idx]
                r200m = r1 + (r2 - r1) * (rho1 - rho_target) / (rho1 - rho2)
                m200m = np.interp(r200m, rps_prof, mass_prof)

        # Compute V200
        v200sq = G_gravity * m200m / r200m

        # Append rescaled quantities
        r.append(rps / r200m)
        vr.append(vrp / np.sqrt(v200sq))
        lnv2.append(np.log(v2p / v200sq))

    # Concatenate into a single array
    r = np.concatenate(r)
    vr = np.concatenate(vr)
    lnv2 = np.concatenate(lnv2)

    return np.vstack([r, vr, lnv2])



def _get_candidate_object_data(
    mini_box_id: int,
    pos_seed: list[np.ndarray],
    vel_seed: list[np.ndarray],
    mass_seed: list[np.ndarray],
    radius_seed: list[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    name: str,
) -> np.ndarray:
    """Extracts all requested seeds from a single minibox.

    Parameters
    ----------
    mini_box_id : int
        Mini-box ID
    pos_seed : list[np.ndarray]
        Seed positions
    vel_seed : list[np.ndarray]
        Seed velocities
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe

    Returns
    -------
    np.ndarray
        Particle's radial distance, radial velocity and log of the square of the
        velocity in units of R200m and M200m.
    """
    # Load particles in minibox.
    pos, vel, *_ = load_objects(mini_box_id, boxsize, minisize, save_path, name=name)

    # Iterate over seeds in current mini box.
    r, vr, lnv2 = ([] for _ in range(3))
    for i in range(len(pos_seed)):
        # Compute the relative positions of all particles in the box
        rel_pos = relative_coordinates(pos, pos_seed[i], boxsize)
        # Only work with those close to the seed
        mask_close = np.prod(np.abs(rel_pos) <= r_max, axis=1, dtype=bool)

        rel_pos = rel_pos[mask_close]
        rel_vel = vel[mask_close] - vel_seed[i]
        
        # Compute radial distance, radial and tangential velocities
        rps = np.sqrt(np.sum(np.square(rel_pos), axis=1))
        vrp, _, v2p = velocity_components(rel_pos, rel_vel)

        r200m = radius_seed[i]
        m200m = mass_seed[i]

        # Compute V200
        v200sq = G_gravity * m200m / r200m
        
        # Append rescaled quantities to containers
        r.append(rps/r200m)
        vr.append(vrp/np.sqrt(v200sq))
        lnv2.append(np.log(v2p/v200sq))
    
    # Concatenate into a single array
    r = np.concatenate(r)
    vr = np.concatenate(vr)
    lnv2 = np.concatenate(lnv2)

    return np.vstack([r, vr, lnv2])



def _select_candidate_seeds(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    name: str = 'part',
    n_threads: int = None,
) -> tuple[np.ndarray]:
    """Locates for the largest `M_200b` seeds and searches for all the particles
    around them up to a distance `r_max`.

    Only seeds that dominate their environment are eligible. This means that the 
    mass of all other seeds up to a distance of 2*R_200b must be at most 20% the
    mass of the seed.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe

    Returns
    -------
    np.ndarray
        Radial distance, radial velocity, and log of the square of the velocity 
        in units of R200m and M200m.
    """
    # Load seed data
    hid, pos_seed, vel_seed, m200b, r200b  = seed_data

    # Rank order by mass.
    order = np.argsort(-m200b)
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]
    r200b = r200b[order]
    m200b = m200b[order]

    # Search for eligible seeds.
    # A seed is considered eligible if it dominates its own environment. This is
    # enforced by requiring that the next most-massive seed within 2*R200 is, at
    # least five times smaller than the seed, i.e. has a mass <= 20% of M200.
    # The loop will stop once it has found `n_seeds` eligible seeds.
    # NOTE: When using multiple threads, this is the part that takes most of the
    # execution time and scales with the number of candidate seeds requested.
    seed_i = []
    print('Looking for candidate seeds...')
    tree = cKDTree(np.mod(pos_seed, boxsize), boxsize=boxsize)
    for i in range(len(pos_seed)):
        idx_close = tree.query_ball_point(pos_seed[i], 2*r200b[i])
        idx_close = [j for j in idx_close if j != i]  
        if np.all(m200b[idx_close] < 0.2 * m200b[i]):
            seed_i.append(i)
        if len(seed_i) >= n_seeds:
            break
    print(f'Found candidate seeds.')

    hid = hid[seed_i]
    pos_seed = pos_seed[seed_i]
    vel_seed = vel_seed[seed_i]
    r200b = r200b[seed_i]
    m200b = m200b[seed_i]

    # Locate mini box IDs for all seeds.
    seed_mini_box_id = get_mini_box_id(pos_seed, boxsize, minisize)
    # Sort by mini box ID
    order = np.argsort(seed_mini_box_id)
    seed_mini_box_id = seed_mini_box_id[order]
    hid = hid[order]
    pos_seed = pos_seed[order]
    vel_seed = vel_seed[order]
    r200b = r200b[order]
    m200b = m200b[order]

    # Get unique mini box ids
    unique_mini_box_ids = np.unique(seed_mini_box_id)
    n_unique = len(unique_mini_box_ids)

    pos_unique = [
        pos_seed[seed_mini_box_id == mini_box_id] 
        for mini_box_id in unique_mini_box_ids
    ]
    vel_unique = [
        vel_seed[seed_mini_box_id == mini_box_id] 
        for mini_box_id in unique_mini_box_ids
    ]
    mass_unique = [
        m200b[seed_mini_box_id == mini_box_id]
        for mini_box_id in unique_mini_box_ids
    ]
    radius_unique = [
        r200b[seed_mini_box_id == mini_box_id]
        for mini_box_id in unique_mini_box_ids
    ]
    
    if name == 'part':
        func = partial(_get_candidate_particle_data, r_max=r_max, boxsize=boxsize, 
                   minisize=minisize, save_path=save_path, part_mass=part_mass, 
                   rhom=rhom)
        data = zip(unique_mini_box_ids, pos_unique, vel_unique)
    elif name == 'gas' or name == 'stars':
        func = partial(_get_candidate_object_data, r_max=r_max, boxsize=boxsize, 
                   minisize=minisize, save_path=save_path, name=name)
        data = zip(unique_mini_box_ids, pos_unique, vel_unique, mass_unique, radius_unique)

    # Cap the number of threads to the total number of miniboxes to process.
    if not n_threads:
        n_threads = np.min([os.cpu_count()-10, n_unique])
    else:
        n_threads = np.min([n_threads, n_unique])

    
    out = []
    with ThreadPoolExecutor(max_workers=n_threads) as executor:
        futures = [executor.submit(func, *args) for args in data]
        for f in tqdm(as_completed(futures), total=len(futures),
                    colour="blue", ncols=100,
                    desc='Processing candidates'):
            out.append(f.result())
    out = np.concatenate(out, axis=1).T

    # Return an array where each column corresponds to r, vr, lnv2 respectively
    return out



def get_calibration_data(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    name: str = 'part',
    n_threads: int = None,
    calibration_save_path: str = None,
    calibration_data_file_name: str = None,
) -> tuple[np.ndarray]:
    """_summary_

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe
    n_threads : int
        Number of threads, by default None

    Returns
    -------
    tuple[np.ndarray]
        Radial distance, radial velocity, and log of the square of the velocity
    """
    if calibration_data_file_name is not None:
        if calibration_save_path is not None:
            file_name = calibration_save_path + calibration_data_file_name
        else:
            file_name = save_path + calibration_data_file_name
    else:
        file_name = save_path + f'calibration_data_{name}.hdf5'
    try:
        with h5.File(file_name, 'r') as hdf:
            r = hdf['r'][()]
            vr = hdf['vr'][()]
            lnv2 = hdf['lnv2'][()]
        return r, vr, lnv2
    except:
        out = _select_candidate_seeds(
            n_seeds=n_seeds,
            seed_data=seed_data,
            r_max=r_max,
            boxsize=boxsize,
            minisize=minisize,
            save_path=save_path,
            part_mass=part_mass,
            rhom=rhom,
            name=name,
            n_threads=n_threads,
        )

        with h5.File(file_name, 'w') as hdf:
            hdf.create_dataset('r', data=out[:, 0])
            hdf.create_dataset('vr', data=out[:, 1])
            hdf.create_dataset('lnv2', data=out[:, 2])

        return out[:, 0], out[:, 1], out[:, 2]


def cost_percentile(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the `target` percentile of particles is below the line.

    Parameters
    ----------
    b : float
        Fit parameter
    *data : tuple
        A tuple with `[r, lnv2, slope, target]`, where `slope` is the slope of 
        the line and is fixed, and `target` is the desired percentile

    Returns
    -------
    float
    """
    r, lnv2, slope, target, r0 = data
    line = slope * (r - r0) + b
    below_line = (lnv2 < line).sum()
    return np.log((target - below_line / r.shape[0]) ** 2)


def cost_perp_distance(b: float, *data) -> float:
    """Cost function for y-intercept b parameter. The optimal value of b is such
    that the perpendicular distance of all points to the line is maximal
    Parameters
    ----------
    b : float
        Fit parameter
    *data: tuple
        A tuple with `[r, lnv2, slope, width]`, where `slope` is the slope of 
        the line and is fixed, and `width` is the width of a band around the 
        line within which the distance is computed
        
    Returns
    -------
    float
    """
    r, lnv2, slope, width, r0 = data
    line = slope * (r - r0) + b
    d = np.abs(lnv2 - line) / np.sqrt(1 + slope**2)
    return -np.log(np.mean(d[(d < width)] ** 2))


def gradient_minima(
    r: np.ndarray,
    lnv2: np.ndarray,
    mask_vr: np.ndarray,
    n_points: int,
    r_min: float,
    r_max: float,
    mask_lnv2: List[np.ndarray] = [1.0, 2.0],
) -> tuple[np.ndarray]:
    """Computes the r-lnv2 gradient and finds the minimum as a function of `r`
    within the interval `[r_min, r_max]`

    Parameters
    ----------
    r : np.ndarray
        Radial separation
    lnv2 : np.ndarray
        Log-kinetic energy
    mask_vr : np.ndarray
        Mask for the selection of radial velocity
    n_points : int
        Number of minima points to compute
    r_min : float
        Minimum radial distance
    r_max : float
        Maximum radial distance

    Returns
    -------
    tuple[np.ndarray]
        Radial and minima coordinates.
    """
    mask_min, mask_max = mask_lnv2
    r_edges_grad = np.linspace(r_min, r_max, n_points + 1)
    grad_r = 0.5 * (r_edges_grad[:-1] + r_edges_grad[1:])
    grad_min = np.zeros(n_points)
    for i in range(n_points):
        r_mask = (r > r_edges_grad[i]) * (r < r_edges_grad[i + 1])
        hist_yv, hist_edges = np.histogram(lnv2[mask_vr * r_mask], bins=200)
        hist_lnv2 = 0.5 * (hist_edges[:-1] + hist_edges[1:])
        hist_lnv2_grad = np.gradient(hist_yv, np.mean(np.diff(hist_edges)))
        lnv2_mask = (mask_min < hist_lnv2) * (hist_lnv2 < mask_max)
        grad_min[i] = hist_lnv2[lnv2_mask][np.argmin(hist_lnv2_grad[lnv2_mask])]

    return grad_r, grad_min


def self_calibration(
    n_seeds: int,
    seed_data: tuple[np.ndarray],
    r_max: float,
    boxsize: float,
    minisize: float,
    save_path: str,
    part_mass: float,
    rhom: float,
    name: str = 'part',
    n_points: int = 20,
    perc: float = 0.995,
    width: float = 0.05,
    grad_lims_pos: tuple[float] = (0.2, 0.5),
    grad_lims_neg: tuple[float] = (0.2, 0.5),
    mask_lnv2_pos: List[np.ndarray] = [1.0, 2.0],
    mask_lnv2_neg: List[np.ndarray] = [1.0, 2.0],
    n_threads: int = None,
    calibration_save_path: str = None,
    calibration_data_file_name: str = None,
    calibration_pars_file_name: str = None,
    whether_pars_compute: bool = True,
) -> None:
    """Runs calibration from isolated halo samples.

    Parameters
    ----------
    n_seeds : int
        Number of seeds to process
    seed_data : tuple[np.ndarray]
        Tuple with seed ID, positions, velocities, M200b and R200b.
    r_max : float
        Maximum distance to consider
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Path to the mini boxes
    part_mass : float
        Mass per particle
    rhom : float
        Mass density of the universe
    n_points : int, optional
        Number of minima points to compute, by default 20
    perc : float, optional
        Target percentile for the positive radial velocity calibration, 
        by default 0.98
    width : float, optional
        Band width for the negattive radial velocity calibration, 
        by default 0.05
    grad_lims : tuple[float]
        Radial interval where the gradient is computed, by default (0.2, 0.5)
    n_threads : int
        Number of threads, by default None
    """
    r, vr, lnv2 = get_calibration_data(
        n_seeds=n_seeds,
        seed_data=seed_data,
        r_max=r_max,
        boxsize=boxsize,
        minisize=minisize,
        save_path=save_path,
        part_mass=part_mass,
        rhom=rhom,
        name=name,
        n_threads=n_threads,
        calibration_save_path=calibration_save_path,
        calibration_data_file_name=calibration_data_file_name
    )
    if not whether_pars_compute:
        return
    


    mask_vr_neg = (vr < 0)
    mask_vr_pos = ~mask_vr_neg
    mask_r = r <= 2.0
    x0 = 0.5

    # For vr > 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_pos, n_points, 
                                       *grad_lims_pos, mask_lnv2=mask_lnv2_pos)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * (x - x0) + b, r_grad, min_grad, 
                        p0=[-1, 2], bounds=((-5, 0), (0, 5)))
    slope_pos, pivot_0 = popt

    # Find intercept by finding the value that contains 'perc' percent of
    # particles below the line at fixed slope 'm_pos'.
    res = minimize(
        fun=cost_percentile,
        x0=1.1 * pivot_0,
        bounds=((pivot_0, 5.0),),
        args=(r[mask_vr_pos&mask_r], lnv2[mask_vr_pos&mask_r], slope_pos, perc, x0),
        method='Nelder-Mead',
    )
    b_pivot_pos = res.x[0]

    # For vr < 0 ===============================================================
    r_grad, min_grad = gradient_minima(r, lnv2, mask_vr_neg, n_points, 
                                       *grad_lims_neg, mask_lnv2=mask_lnv2_neg)
    # Find slope by fitting to the minima.
    popt, _ = curve_fit(lambda x, m, b: m * (x - x0) + b, r_grad, min_grad, 
                        p0=[-1, 2], bounds=((-5, 0), (0, 3)))
    slope_neg, pivot_1 = popt

    # Find intercept by finding the value that maximizes the perpendicular
    # distance to the line at fixed slope of all points within a perpendicular
    # 'width' distance from the line (ignoring all others).
    res = minimize(
        fun=cost_perp_distance,
        x0=0.8 * pivot_1,
        bounds=((0.5 * pivot_1, pivot_1),),
        args=(r[mask_vr_neg], lnv2[mask_vr_neg], slope_neg, width, x0),
        method='Nelder-Mead',
    )
    b_pivot_neg = res.x[0]
    
    b_neg = b_pivot_neg - slope_neg * x0
    gamma = 2.
    alpha = (gamma - b_neg) / x0**2
    beta = slope_neg - 2 * alpha * x0
    

    if calibration_pars_file_name is not None:
        if calibration_save_path is not None:
            file_name = calibration_save_path + calibration_pars_file_name
        else:
            file_name = save_path + calibration_pars_file_name
    else:
        file_name = save_path + f'calibration_pars_{name}.hdf5'
    with h5.File(file_name, 'w') as hdf:
        hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
        hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
        hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])

    return


def calibrate(
    save_path: str, 
    pos_neg: np.array = None,
    omega_m: float = None, 
    name: str = 'part',
    **kwargs,
) -> None:
    """Calibrates finder by assuming cosmology dependence. If `omega_m` is 
    `None`, then it runs the calibration on the simulation data directly.

    Parameters
    ----------
    save_path : str
        Path to the mini boxes. Saves the calibration parameter in this directory.
    omega_m : float, optional
        Matter density parameter Omega matter, by default None.
    **kwargs
        See `run_calibrate` for parameter descriptions.
    """
    if omega_m:
        slope_pos = -1.915
        b_pivot_pos = 1.664 + 0.74 * (omega_m - 0.3)

        x0 = 0.5
        slope_neg = -1.592 + 0.696 * (omega_m - 0.3)
        b_pivot_neg = 0.8 + 0.525 * (omega_m - 0.3)
        b_neg = b_pivot_neg - slope_neg * x0
        gamma = 2.
        alpha = (gamma - b_neg) / x0**2
        beta = slope_neg - 2 * alpha * x0

        with h5.File(save_path + f'calibration_pars_{name}.hdf5', 'w') as hdf:
            hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
            hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
            hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])
    else:
        if pos_neg is not None:
            slope_pos, b_pivot_pos = pos_neg[0]
            slope_neg, b_pivot_neg = pos_neg[1]
            x0 = 0.5
            b_neg = b_pivot_neg - slope_neg * x0
            gamma = 2.
            alpha = (gamma - b_neg) / x0**2
            beta = slope_neg - 2 * alpha * x0

            with h5.File(save_path + f'calibration_pars_{name}.hdf5', 'w') as hdf:
                hdf.create_dataset('pos', data=[slope_pos, b_pivot_pos])
                hdf.create_dataset('neg/line', data=[slope_neg, b_pivot_neg])
                hdf.create_dataset('neg/quad', data=[alpha, beta, gamma])
        else:
            self_calibration(save_path=save_path, name=name, **kwargs)

    return


if __name__ == "__main__":
    pass
