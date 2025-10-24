import os
from functools import partial
from multiprocessing import Pool
from warnings import filterwarnings

import h5py as h5
import numpy as np
import pandas as pd
from scipy.optimize import fsolve
from tqdm import tqdm
from scipy.spatial import cKDTree
from numba import njit


from oasis.common import G_gravity, mkdir
from oasis.coordinates import relative_coordinates
from oasis.minibox import load_particles, load_seeds

filterwarnings('ignore')


def characteristic_density(
    r200: float,
    rs: float
) -> float:
    """Computes the characteristic density of an NFW profile.

    Parameters
    ----------
    r200 : float
        Halo radius
    rs : float
        Scale radius

    Returns
    -------
    float
        Delta characteristic

    Raises
    ------
    ZeroDivisionError
        If `r200` or `rs` are zero.
    """
    if np.any(rs == 0.) or np.any(r200 == 0.):
        raise ZeroDivisionError('Neither r200 nor rs can be zero.')

    c200 = r200 / rs
    delta = (200./3.) * (c200 ** 3 / (np.log(1 + c200) - (c200 / (1 + c200))))

    return delta


def rho_nfw_roots(
    x: float,
    delta1: float,
    rs1: float,
    delta2: float,
    rs2: float,
    r12: float,
) -> float:
    """Returns the value of 

    \begin{equation*}
        \frac{\rho_1(R-r)}{\rho_{c}} &= \frac{\rho_2(r)}{\rho_{c}}
    \end{equation*}

    where $\rho(r)$ is the NFW profile.

    \begin{equation*}
        \frac{\rho(r)}{\rho_{c}} = 
            \frac{\delta_c}{\frac{r}{R_s}\left(1+\frac{r}{R_s}\right)^2}
    \end{equation*}

    Parameters
    ----------
    x : float
        Radial coordinate
    delta1 : float
        Characteristic density of the central object
    rs1 : float
        Scale radius of the central object
    delta2 : float
        Characteristic density of the substructure
    rs2 : float
        Scale radius of the substructure
    r12 : float
        Radial separation between central and substructure R=|x2-x1|.

    Returns
    -------
    float

    """
    x1 = (r12 - x) / rs1
    x2 = x / rs2
    frac1 = delta1 / (x1 * (1 + x1)**2)
    frac2 = delta2 / (x2 * (1 + x2)**2)
    return frac1 - frac2


@njit
def classify(
    rel_pos: np.ndarray,
    rel_vel: np.ndarray,
    r200: float,
    m200: float,
    class_pars: list | tuple | np.ndarray,
    max_radius: float = 2.0,
    pivot_radius: float = 0.5
) -> np.ndarray:
    """Classifies particles as orbiting.

    Parameters
    ----------
    rel_pos : np.ndarray
        Relative position of particles around seed
    rel_vel : np.ndarray
        Relative velocity of particles around seed
    r200 : float
        Seed R200
    m200 : float
        Seed M200
    class_pars : Union[List, Tuple, np.ndarray]
        Classification parameters [m_pos, b_pos, m_neg, b_neg]
    max_radius : float
        Maximum radius where orbiting particles can be found. All particles 
        above this value are set to be infalling. By default 2.0.
    pivot_radius : float
        Pivot value for the cut transition from linear to quadratic. By default
        0.5.

    Returns
    -------
    np.ndarray
        A boolean array where True == orbiting
    """
    m_pos, b_pos, m_neg, b_neg, alpha, beta, gamma = class_pars
    # Compute V200
    v200 = G_gravity * m200 / r200

    # Compute the radius to seed_i in r200 units, and ln(v^2) in v200 units
    part_ln_vel = np.log(np.sum(np.square(rel_vel), axis=1) / v200)
    part_radius = np.sqrt(np.sum(np.square(rel_pos), axis=1)) / r200

    # Create a mask for particles with positive radial velocity
    mask_vr_positive = np.sum(rel_vel * rel_pos, axis=1) > 0

    # Orbiting classification for vr > 0
    line = m_pos * (part_radius - pivot_radius) + b_pos
    mask_cut_pos = part_ln_vel < line

    # Orbiting classification for vr < 0
    mask_small_radius = part_radius <= pivot_radius
    curve = alpha * part_radius ** 2 + beta * part_radius + gamma
    line = m_neg * (part_radius - pivot_radius) + b_neg

    mask_cut_neg = ((part_ln_vel < curve) & mask_small_radius) ^ \
        ((part_ln_vel < line) & ~mask_small_radius)

    # Particle is infalling if it is below both lines and 2*R00
    mask_orb = (part_radius <= max_radius) & (
        (mask_cut_pos & mask_vr_positive) ^
        (mask_cut_neg & ~mask_vr_positive)
    )

    return mask_orb




def classify_single_mini_box(
    mini_box_id: int,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    run_name: str,
    padding: float = 5.0,
    fast_mass: bool = False,
    part_mass: float = None,
    disable_tqdm: bool = False,
    seed_range: tuple = None,
) -> None:
    """Runs the classifier for each seed in a mini box...

    Parameters
    ----------
    mini_box_id : int
        Mini-box ID
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    load_path : str
        Location from where to load the file
    run_name : str
        Label for the current run. The directory created will be `run_<run_name>`
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    None
    """
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'
    mkdir(save_path)

    # Load seeds in mini box
    pos_seed, vel_seed, hid, r200b, m200b, rs, mask_mb = \
        load_seeds(mini_box_id, boxsize, minisize, load_path, padding)
    
    if seed_range is not None:
        mask_range = (m200b >= seed_range[0]) & (m200b < seed_range[1])
        pos_seed = pos_seed[mask_range]
        vel_seed = vel_seed[mask_range]
        hid = hid[mask_range]
        r200b = r200b[mask_range]
        m200b = m200b[mask_range]
        rs = rs[mask_range]
        mask_mb = mask_mb[mask_range]

    if fast_mass and part_mass:
        min_mass = 0.5 * min_num_part * part_mass
        m200b_mask = m200b > min_mass
        pos_seed = pos_seed[m200b_mask]
        vel_seed = vel_seed[m200b_mask]
        hid = hid[m200b_mask]
        r200b = r200b[m200b_mask]
        m200b = m200b[m200b_mask]
        rs = rs[m200b_mask]
        mask_mb = mask_mb[m200b_mask]
    elif fast_mass and not part_mass:
        raise ValueError('Particle mass unspecified. Please run again with ' +
                         'part_mass argument specified.')

    n_seeds = len(hid)

    # Exit if there are no seeds in the mini box.
    if not any(hid):
        return None
 
    # Concentration parameter and characteristic density
    deltac = characteristic_density(r200b, rs)

    # Load particles
    pos_part, vel_part, pid = \
        load_particles(mini_box_id, boxsize, minisize, load_path, padding)

    # Load calibration parameters
    with h5.File(load_path + 'calibration_pars_part.hdf5', 'r') as hdf:
        pars = (*hdf['pos'][()], *hdf['neg/line'][()], *hdf['neg/quad'][()])

    col_names = (
        'Halo_ID', 'M200b', 'R200b', 'pos', 'vel', 'Norb', 'LIDX', 'RIDX',
        'INMB', 'cm', 'NSUBS', 'PID', 'SLIDX', 'SRIDX'
    )
    haloes = pd.DataFrame(columns=col_names)

    tree = cKDTree(np.mod(pos_part, boxsize), boxsize=boxsize)
    
    orb_pid, orb_hid = [], []
    n_tot_p, n_tot_s = 0, 0
    parent_id_seed = np.full(n_seeds, -1, dtype=int)

    rel_pos_seed_all = pos_seed[:, None, :] - pos_seed[None, :, :]
    rel_pos_seed_all -= np.rint(rel_pos_seed_all / boxsize) * boxsize

    # compute all indices within 2*r200b for each seed using KDTree
    idx_all = tree.query_ball_point(pos_seed, r=2*r200b, p=np.inf)

    for i in tqdm(range(n_seeds), ncols=100, desc='Finding haloes', colour='green', disable=disable_tqdm):
        if parent_id_seed[i] != -1:
            continue

        # ------------------------------
        # Classify particles
        # ------------------------------
        pos_seed_i = pos_seed[i]
        vel_seed_i = vel_seed[i]
        r200b_i = r200b[i]
        m200b_i = m200b[i]

        # get indices of particles of this halo
        idx = idx_all[i]
        pos_part_seed = pos_part[idx]
        vel_part_seed = vel_part[idx]

        rel_pos_all = relative_coordinates(pos_part_seed, pos_seed_i, boxsize)
        rel_vel_all = vel_part_seed - vel_seed_i


        mask_orb = classify(rel_pos_all, rel_vel_all, r200b_i, m200b_i, pars)

        # Check if enough orbiting particles to be a halo
        is_halo = mask_orb.sum() >= min_num_part
        if not is_halo:
            continue

        # ------------------------------
        # Classify seeds
        # ------------------------------
        rel_pos_seed = rel_pos_seed_all[i]  # relative positions to other seeds
        r_max = 2.0 * r200b_i
        mask_seed = (np.sum(rel_pos_seed[i+1:]**2, axis=1) <= r_max**2) & \
                    (parent_id_seed[i+1:] == -1)

        n_seeds_near = mask_seed.sum()
        orb_seed = []
        if n_seeds_near > 0:
            # Precompute information for nearby seeds
            pos_seed_near = pos_seed[i+1:][mask_seed]
            vel_seed_near = vel_seed[i+1:][mask_seed]
            deltac_seed_near = deltac[i+1:][mask_seed]
            rs_seed_near = rs[i+1:][mask_seed]
            rel_pos_seed_near = rel_pos_seed[i+1:][mask_seed]
            r_ij_all = np.linalg.norm(rel_pos_seed_near, axis=1)

         
            delta_pos = pos_part_seed[:, None, :] - pos_seed_near[None, :, :]
            delta_pos -= np.rint(delta_pos / boxsize) * boxsize
            delta_vel = vel_part_seed[:, None, :] - vel_seed_near[None, :, :]
            rp_sq_all = np.sum(delta_pos**2, axis=2)
            vp_sq_all = np.sum(delta_vel**2, axis=2)

            orb_seed = []
            if fast_mass:
                rel_vel_seed = vel_seed - vel_seed_i
                mask_orb_sub = classify(rel_pos_seed[i+1:][mask_seed],
                                        rel_vel_seed[i+1:][mask_seed], r200b_i,
                                        m200b_i, pars)
                if mask_orb_sub.sum() > 0:
                    orb_seed.extend(hid[i+1:][mask_seed][mask_orb_sub])
            else:
                j = 0
                while is_halo and (j < n_seeds_near):
                    # Extract radial and velocity squared arrays
                    rp_sq = rp_sq_all[:, j]
                    vp_sq = vp_sq_all[:, j]
                    r_ij = r_ij_all[j]

                    r_ball = fsolve(
                        func=rho_nfw_roots,
                        x0=0.5*r_ij,
                        args=(deltac[i], rs[i], deltac_seed_near[j], rs_seed_near[j], r_ij)
                    )[0]
                    r_ball = min(r_ball, r200b[i+1:][mask_seed][j])

                    v_ball_sq = 4.0 * G_gravity * m200b[i+1:][mask_seed][j] / \
                                r200b[i+1:][mask_seed][j]

                    ball6d = (rp_sq <= r_ball**2) & (vp_sq <= v_ball_sq)
                    frac_inside = (ball6d * mask_orb).sum() / ball6d.sum()

                    f_threshold = max(0.5, 1. - np.exp(-(r_ij/r200b[i+1:][mask_seed][j])**2))
                    if frac_inside >= f_threshold:
                        orb_seed.append(hid[i+1:][mask_seed][j])
                        mask_orb[ball6d] = True
                    else:
                        mask_orb[ball6d] = False

                    is_halo = mask_orb.sum() >= min_num_part
                    j += 1

        if not is_halo:
            continue

        # Set parent halo ID for seeds (these are no longer free).
        mask_subs = np.isin(hid[i+1:], orb_seed)
        parent_id_seed[i+1:][mask_subs] = hid[i]
        n_subs = mask_subs.sum()

        # Append halo to catalogue =============================================
        n_orb = mask_orb.sum()
        haloes.loc[len(haloes.index)] = [
            hid[i],
            m200b[i],
            r200b[i],
            pos_seed[i],
            vel_seed[i],
            n_orb,
            n_tot_p,
            n_tot_p + n_orb,
            mask_mb[i],
            pos_seed[i] + np.mean(rel_pos_all[mask_orb], axis=0),
            n_subs,
            parent_id_seed[i],
            n_tot_s,
            n_tot_s + len(orb_seed),
        ]
        n_tot_p += n_orb
        n_tot_s += len(orb_seed)

        orb_pid.append(pid[idx][mask_orb])
        orb_hid.append(orb_seed)

       
  
    orb_pid = np.concatenate(orb_pid)
    orb_hid = np.concatenate(orb_hid)

    Halo_ID = haloes['Halo_ID'].to_numpy()
    M200b = haloes['M200b'].to_numpy()
    R200b = haloes['R200b'].to_numpy()
    pos = np.stack(haloes['pos'].to_numpy())
    vel = np.stack(haloes['vel'].to_numpy())
    cm = np.stack(haloes['cm'].to_numpy())
    NSUBS = haloes['NSUBS'].to_numpy()
    PID = haloes['PID'].to_numpy()
    LIDX = haloes['LIDX'].to_numpy()
    RIDX = haloes['RIDX'].to_numpy()
    SLIDX = haloes['SLIDX'].to_numpy()
    SRIDX = haloes['SRIDX'].to_numpy()
    INMB = haloes['INMB'].to_numpy()

    # Preallocate result containers
    orb_pid_perc = []
    orb_hid_perc = []
    halo_records = []

    n_tot_perc = 0
    n_tot_s_perc = 0

    # Use sets to speed up isin queries
    orb_pid_seen = set()
    orb_hid_seen = set()

    for i in tqdm(range(len(haloes)), ncols=100, desc='Saving particles (fast)',
                    colour='green', disable=disable_tqdm):

        # if not INMB[i]:
        #     continue

        lidx, ridx = LIDX[i], RIDX[i]
        slidx, sridx = SLIDX[i], SRIDX[i]

        # Use sets to speed up isin queries
        pid_range = orb_pid[lidx:ridx]
        new_orb = [p for p in pid_range if p not in orb_pid_seen]
        orb_pid_seen.update(new_orb)

        hid_range = orb_hid[slidx:sridx]
        new_orb_s = [h for h in hid_range if h not in orb_hid_seen]
        orb_hid_seen.update(new_orb_s)

        n_orb = len(new_orb)
        n_orb_s = len(new_orb_s)

        if (n_orb >= min_num_part) and INMB[i]:
            orb_pid_perc.append(np.array(new_orb, dtype=pid.dtype))
            orb_hid_perc.append(np.array(new_orb_s, dtype=hid.dtype))

            halo_records.append((
                Halo_ID[i], M200b[i], R200b[i], pos[i], vel[i],
                n_orb * part_mass, n_orb,
                n_tot_perc, n_tot_perc + n_orb,
                cm[i], NSUBS[i], PID[i],
                n_tot_s_perc, n_tot_s_perc + n_orb_s
            ))

            n_tot_perc += n_orb
            n_tot_s_perc += n_orb_s

    # Concatenate particle and seed arrays
    orb_pid_perc = np.concatenate(orb_pid_perc)
    orb_hid_perc = np.concatenate(orb_hid_perc)

    # Create halo dataframe
    col_names = (
        'Halo_ID', 'M200b', 'R200b', 'pos', 'vel', 'Morb', 'Norb',
        'LIDX', 'RIDX', 'cm', 'NSUBS', 'PID', 'SLIDX', 'SRIDX'
    )
    haloes_perc = pd.DataFrame(halo_records, columns=col_names)


    dtypes = (
        np.uint32, np.float32, np.float32, np.float32, np.float32, np.float32,
        np.uint32, np.uint32, np.uint32, np.float32, np.uint16, np.int32,
        np.uint32, np.uint32
    )

    with h5.File(f'{save_path}/{mini_box_id}.hdf5', 'w') as hdf:
        # Halo catalogue
        for i, key in enumerate(col_names):
            values = haloes_perc[key].values
            data = np.stack(values) if key in ['pos', 'vel', 'cm'] else values
            hdf.create_dataset(f'halo/{key}', data=data, dtype=dtypes[i])

        # Particles
        hdf.create_dataset('memb/PID', data=orb_pid_perc, dtype=pid.dtype)

        # Seeds
        hdf.create_dataset('memb/Halo_ID', data=orb_hid_perc, dtype=hid.dtype)


    return None

    

def run_orbiting_mass_assignment(
    load_path: str,
    run_name: str,
    min_num_part: int,
    boxsize: float,
    minisize: float,
    padding: float,
    fast_mass: bool = False,
    part_mass: float = None,
    n_threads: int = None,
    cleanup: bool = False,
    seed_range: tuple = None
) -> None:
    """Generates a halo catalogue using the kinetic mass criterion to classify
    particles into orbiting or infalling.

    Parameters
    ----------
    load_path : str
        Location from where to load the file
    min_num_part : int
        Minimum number of particles needed to be considered a halo
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    run_name : str
        Label for the current run. The directory created will be `run_<run_name>`
    padding : float, optional
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5
    n_threads : int
        Number of threads, by default None
    cleanup : bool
        Removes individual minibox catalogues after contatenation.

    Returns
    -------
    None
    """
    # Create directory if it does not exist
    save_path = load_path + f'run_{run_name}/mini_box_catalogues/'
    mkdir(save_path)

    # Number of miniboxes
    n_mini_boxes = np.int_(np.ceil(boxsize / minisize))**3
    n_threads = np.min([n_threads, n_mini_boxes])

    # Parallel processing of miniboxes.
    func = partial(classify_single_mini_box, min_num_part=min_num_part,
                   boxsize=boxsize, load_path=load_path, minisize=minisize,
                   padding=padding, run_name=run_name, fast_mass=fast_mass,
                   part_mass=part_mass, disable_tqdm=True, seed_range=seed_range)
    
    with Pool(n_threads) as pool, \
        tqdm(total=n_mini_boxes, colour="green", ncols=100,
             desc='Generating halo catalogue') as pbar:
        for _ in pool.imap(func, range(n_mini_boxes)):
            pbar.update()
            pbar.refresh()

    n_part, n_seed = 0, 0
    first_file = True

    with h5.File(save_path + f'0.hdf5', 'r') as hdf_load:
        halo_keys = list((hdf_load['halo'].keys()))
    halo_data = [[] for _ in range(len(halo_keys))]

    hdf_memb = h5.File(load_path + f'run_{run_name}/members.hdf5', 'w')
    for i in tqdm(range(n_mini_boxes), ncols=100, desc='Merging catalogues',
                  colour='green'):
        with h5.File(save_path + f'{i}.hdf5', 'r') as hdf_load:
            if 'halo' not in hdf_load.keys():
                continue
            # Member data ======================================================
            # Number of particles in current file
            n_part_this = hdf_load['memb/PID'].shape[0]
            n_seed_this = hdf_load['memb/Halo_ID'].shape[0]

            # This reshaping of the dataset after every new file...
            if first_file:  # Create the dataset at first pass.
                hdf_memb.create_dataset(name='PID',
                                        chunks=True, maxshape=(None,),
                                        data=hdf_load['memb/PID'][()])
                hdf_memb.create_dataset(name='Halo_ID',
                                        chunks=True, maxshape=(None,),
                                        data=hdf_load['memb/Halo_ID'][()])
                first_file = False
            else:
                # Number of particles so far plus this file's total.
                new_shape = n_part + n_part_this
                # Resize axes and save incoming data
                hdf_memb['PID'].resize((new_shape), axis=0)
                hdf_memb['PID'][n_part:] = hdf_load['memb/PID'][()]

                # Number of seeds so far plus this file's total.
                new_shape = n_seed + n_seed_this
                # Resize axes and save incoming data
                hdf_memb['Halo_ID'].resize((new_shape), axis=0)
                hdf_memb['Halo_ID'][n_seed:] = \
                    hdf_load['memb/Halo_ID'][()]

            # Halo data ========================================================
            for k, key in enumerate(halo_keys):
                if key in ['LIDX', 'RIDX']:
                    halo_data[k].append(hdf_load[f'halo/{key}'][()] + n_part)
                elif key in ['SLIDX', 'SRIDX']:
                    halo_data[k].append(hdf_load[f'halo/{key}'][()] + n_seed)
                else:
                    halo_data[k].append(hdf_load[f'halo/{key}'][()])

            # Add the total number of particles in this file to the next.
            n_part += n_part_this
            n_seed += n_seed_this

    hdf_memb.close()

    # Set the seed index to -1 for all those haloes without subhaloes.
    for k, key in enumerate(halo_keys):
        if key == 'SLIDX':
            slidx = np.concatenate(halo_data[k], dtype=np.int32)
        elif key == 'SRIDX':
            sridx = np.concatenate(halo_data[k], dtype=np.int32)

    mask = (sridx-slidx == 0)
    slidx[mask] = -1
    sridx[mask] = -1

    with h5.File(load_path + f'run_{run_name}/catalogue.hdf5', 'w') as hdf:
        hdf.create_dataset('SLIDX', data=slidx)
        hdf.create_dataset('SRIDX', data=sridx)
        for k, key in enumerate(halo_keys):
            if key in ['SLIDX', 'SRIDX']:
                continue
            hdf.create_dataset(key, data=np.concatenate(halo_data[k]))

    if cleanup:
        path = load_path + f'run_{run_name}/mini_box_catalogues/'
        for item in os.listdir(path):
            os.remove(path + item)
        os.removedirs(path)

    return None


if __name__ == '__main__':
    pass
