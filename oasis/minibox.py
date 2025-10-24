from typing import Tuple
import os

import h5py as h5
import numpy as np
from tqdm import tqdm
from os.path import join

from joblib import Parallel, delayed
from glob import glob
import swiftsimio as sw
import shutil


from unyt import Msun, Mpc, km, s

from oasis.common import get_np_unit_dytpe, mkdir
from oasis.coordinates import (cartesian_product, gen_data_pos_regular,
                               relative_coordinates)


def generate_mini_box_grid(
    boxsize: float,
    minisize: float,
) -> Tuple[np.ndarray]:
    """Generates a 3D grid of mini boxes.

    Parameters
    ----------
    boxsize : float
        Size of simulation box.
    minisize : float
        Size of mini box.

    Returns
    -------
    Tuple[np.ndarray]
        ID and central coordinate for all mini boxes.
    """

    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Sub-box central coordinate. Populate each mini box with one point at the
    # centre.
    centres = gen_data_pos_regular(boxsize, minisize)

    # Shift in each dimension for numbering mini boxes
    uint_dtype = get_np_unit_dytpe(boxes_per_side**2)
    shift = np.array([1, boxes_per_side, boxes_per_side**2], dtype=uint_dtype)

    # Set of all possible unique IDs for each mini box
    n = np.arange(boxes_per_side, dtype=uint_dtype)
    ids = np.sum(np.int_(cartesian_product([n, n, n])) * shift, axis=1)
    sort_order = np.argsort(ids)

    # Sort IDs so that the ID matches the row index.
    ids = ids[sort_order]
    centres = centres[sort_order]

    return ids, centres


def get_mini_box_id(
    x: np.ndarray,
    boxsize: float,
    minisize: float,
) -> int:
    """Returns the mini box ID to which the coordinates `x` fall into

    Parameters
    ----------
    x : np.ndarray
        Position in cartesian coordinates.
    boxsize : float
        Size of simulation box.
    minisize : float
        Size of mini box.

    Returns
    -------
    int
        ID of the mini box.
    """
    # periodic boundary conditions
    x = np.mod(x, boxsize)
    
    # Number of mini boxes per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    # Determine data type for integer arrays based on the maximum number of
    # elements
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    # Shift in each dimension for numbering mini boxes
    shift = np.array(
        [1, boxes_per_side, boxes_per_side * boxes_per_side], dtype=uint_dtype)
    # In the rare case an object is located exactly at the edge of the box,
    # move it 'inwards' by a tiny amount so that the box id is correct.
    # x[np.where(x == boxsize)] -= 1e-8
    # x[np.where(x == 0)] += 1e-8

    id = np.floor(x / minisize)
    id[id == boxes_per_side] -= 1
    
    if x.ndim > 1:
        return np.int_(np.sum(shift * id, axis=1))
    else:
        return np.int_(np.sum(shift * id))


def get_adjacent_mini_box_ids(
    mini_box_id: np.ndarray,
    boxsize: float,
    minisize: float,
) -> np.ndarray:
    """Returns a list of all IDs that are adjacent to the specified mini box ID.
    There are always 27 adjacent boxes in a 3D volume, including the specified 
    ID.

    Parameters
    ----------
    mini_box_id : np.ndarray
        ID of the mini box.
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box

    Returns
    -------
    np.ndarray
        List of mini box IDs adjacent to `id`

    Raises
    ------
    ValueError
        If `id` is not found in the allowed values in `ids`
    """
    mini_box_ids, centres = generate_mini_box_grid(boxsize, minisize)
    if mini_box_id not in mini_box_ids:
        raise ValueError(f'ID {mini_box_id} is out of bounds')

    x0 = centres[mini_box_ids == mini_box_id]
    radius = relative_coordinates(centres, x0, boxsize)
    radius = np.sqrt(np.sum(np.square(radius), axis=1))
    mask = radius <= 1.01 * np.sqrt(3.) * minisize
    return mini_box_ids[mask]


def generate_mini_box_ids(
    positions: np.ndarray,
    boxsize: float,
    minisize: float,
    chunksize: int = 100_000,
    tqdm_disable: bool = False,
) -> None:
    """Gets the mini box ID for each position

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    save_path : str
        Where to save the IDs
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None

    Returns
    -------
    None
    """
    n_items = positions.shape[0]
    n_iter = n_items // chunksize

    # Determine data type for integer arrays based on the maximum number of
    # elements
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    uint_dtype = get_np_unit_dytpe(boxes_per_side**3)

    ids = np.zeros(n_items, dtype=uint_dtype)

    for chunk in tqdm(range(n_iter), desc='Chunk', ncols=100, colour='blue', disable=tqdm_disable):
        low = chunk * chunksize
        if chunk < n_iter - 2:
            upp = (chunk + 1) * chunksize
        else:
            upp = None
        ids[low:upp] = get_mini_box_id(positions[low:upp], boxsize, minisize)

    return ids


def get_chunks(ids: np.ndarray, chunksize: int) -> np.ndarray:
    """_summary_

    Parameters
    ----------
    ids : np.ndarray
        _description_
    chunksize : int
        _description_

    Returns
    -------
    np.ndarray
        _description_

    Raises
    ------
    RuntimeError
        _description_
    """
    n_items = ids.shape[0]

    # split in chunksize
    chunks = list(range(0, n_items, chunksize))

    # Ensure the last chunk covers the remaining particles
    if chunks[-1] != n_items:
        chunks.append(n_items)

    return np.array(chunks)
    


def split_box_into_mini_boxes(
    positions: np.ndarray,
    velocities: np.ndarray,
    uid: np.ndarray,
    save_path: str,
    boxsize: float,
    minisize: float,
    chunksize: int = 100_000,
    name: str = None,
    props: Tuple[list, list, list] = None,
    subset: int = None,
) -> None:
    """Sorts all items into mini boxes and saves them in disc.

    Parameters
    ----------
    positions : np.ndarray
        Cartesian coordinates
    velocities : np.ndarray
        Cartesian velocities
    uid : np.ndarray
        Unique IDs for each position (e.g. PID, HID)
    save_path : str
        Where to save the IDs
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    chunksize : int, optional
        Number of items to process at a time in chunks, by default 100_000
    name : str, optional
        An additional name or identifier appended at the end of the file name, 
        by default None
    props : tuple[list(array), list(str), list(dtype)], optional
        Additional arrays to be sorted into mini boxes.

    Returns
    -------
    None

    Raises
    ------
    RuntimeError
        If `chunksize` is too small, the chunk-finding loop cannot properly 
        resolve all the mini box ids within a chunk.
    """
    tqdm_disable = False
    if subset is not None:
        tqdm_disable = True
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # If given chunk size is smaller than the number of points given.
    chunksize = np.min([len(positions), chunksize])
    
    # Compute mini box ids
    mini_box_ids = generate_mini_box_ids(
        positions=positions,
        boxsize=boxsize,
        minisize=minisize,
        chunksize=chunksize,
        tqdm_disable=tqdm_disable,
    )

    # Sort data by mini box id
    mb_order = np.argsort(mini_box_ids)
    mini_box_ids = mini_box_ids[mb_order]
    velocities = velocities[mb_order]
    positions = positions[mb_order]
    uid = uid[mb_order]

    if props:
        labels = props[1]
        dtypes = props[2]
        props = props[0]
        for k, item in enumerate(props):
            props[k] = item[mb_order]

    chunk_idx = get_chunks(
        ids=mini_box_ids,
        chunksize=chunksize
    )

    # Get smallest data type to represent IDs
    uint_dtype_pid = get_np_unit_dytpe(np.max(uid))
    if props:
        labels = ('ID', 'pos', 'vel', *labels)
        dtypes = (uint_dtype_pid, np.float32, np.float32, *dtypes)
    else:
        labels = ('ID', 'pos', 'vel')
        dtypes = (uint_dtype_pid, np.float32, np.float32)

    # Create target directory
    if subset is None:
        save_dir = save_path + f'mini_boxes_nside_{boxes_per_side}/'
    else:
        save_dir = save_path + f'subset_{subset}/'
    mkdir(save_dir)

    if name:
        total_mini_boxes = boxes_per_side ** 3
        for mb_id in range(total_mini_boxes):
            file_path = save_dir + f'{mb_id}.hdf5'
            if os.path.exists(file_path):
                with h5.File(file_path, 'a') as hdf:
                    if name in hdf:
                        del hdf[name]

    # For each chunk

    for chunk_i in tqdm(range(len(chunk_idx)-1), desc='Processing chunks',
                        ncols=100, colour='blue', disable=tqdm_disable):
        # Select chunk
        low = chunk_idx[chunk_i]
        upp = chunk_idx[chunk_i + 1]

        mb_chunk = mini_box_ids[low: upp]
        pos_chunk = positions[low: upp]
        vel_chunk = velocities[low: upp]
        pid_chunk = uid[low: upp]

        if props:
            props_chunks = [item[low: upp] for item in props]

        # Find unique mini box ids in this chunk
        unique_mb_ids = np.unique(mb_chunk)

        for mb_id in unique_mb_ids:
            # Find the particle indices in the current mini-box
            idx = np.where(mb_chunk == mb_id)[0]

            if len(idx) == 0:
                continue  # Prevent empty slices

            start, end = idx[0], idx[-1]+1  # Note that the right boundary of the slice should be +1

            if props:
                data = (
                    pid_chunk[start:end],
                    pos_chunk[start:end],
                    vel_chunk[start:end],
                    *[item_chunk[start:end] for item_chunk in props_chunks]
                )
            else:
                data = (
                    pid_chunk[start:end],
                    pos_chunk[start:end],
                    vel_chunk[start:end]
                )

            with h5.File(save_dir + f'{mb_id}.hdf5', 'a') as hdf:
                prefix = f'{name}/' if name else ''
                for label_i, data_i, dtype_i in zip(labels, data, dtypes):
                    full_name = prefix + f'{label_i}'
                    if full_name in hdf:
                        # dataset already exists, append data
                        dset = hdf[full_name]
                        old_len = dset.shape[0]
                        new_len = old_len + data_i.shape[0]
                        dset.resize((new_len, *dset.shape[1:]))  # expand dataset
                        dset[old_len:] = data_i                  # write new data
                    else:
                        # dataset does not exist, create expandable dataset
                        maxshape = (None, *data_i.shape[1:])    # first dimension is unlimited
                        hdf.create_dataset(full_name, data=data_i, dtype=dtype_i,
                                        maxshape=maxshape, chunks=True)

    return None


def _load_one(file_path, name: str = "part"):
    with h5.File(file_path, 'r') as hdf:
        if name == "part":
            return (
                hdf['part/pos'][:],
                hdf['part/vel'][:],
                hdf['part/ID'][:]
            )
        elif name == "gas":
            return (
                hdf['gas/pos'][:],
                hdf['gas/vel'][:],
                hdf['gas/ID'][:],
                hdf['gas/Mass'][:],
            )
        elif name == "stars":
            return (
                hdf['stars/pos'][:],
                hdf['stars/vel'][:],
                hdf['stars/ID'][:],
                hdf['stars/Mass'][:],
            )
        
def load_particles(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
    parallel: bool = True,
)-> Tuple[np.ndarray]:
    """Load particles from a mini box including all particles in adjacent boxes 
    up to the `padding` distance.

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    load_path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5.

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, and PID
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    )

    # Load all adjacent boxes
    if not parallel:
        # Create empty lists (containers) to save the data from file for each ID
        pos, vel, pid = ([] for _ in range(3))

        # Load all adjacent boxes
        for i, mini_box in enumerate(mini_box_ids):
            file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box}.hdf5'
            with h5.File(load_path + file_name, 'r') as hdf:
                pos.append(hdf['part/pos'][()])
                vel.append(hdf['part/vel'][()])
                pid.append(hdf['part/ID'][()])

        # Concatenate into a single array
        pos = np.concatenate(pos)
        vel = np.concatenate(vel)
        pid = np.concatenate(pid)

        # Mask particles within a padding distance of the edge of the box in each
        # direction
        loc_id = grid_ids == mini_box_id
        padded_distance = 0.5 * minisize + padding
        absolute_rel_pos = np.abs(
            relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
        )
        mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

        return pos[mask], vel[mask], pid[mask]
    else:
        file_paths = [
            os.path.join(load_path, f'mini_boxes_nside_{boxes_per_side}', f'{bid}.hdf5')
            for bid in mini_box_ids
        ]

        results = Parallel(n_jobs=len(file_paths))(
            delayed(_load_one)(fp, name="part") for fp in file_paths
        )

        pos, vel, pid = map(np.concatenate, zip(*results))

        # Mask particles within a padding distance of the edge of the box in each
        # direction

        loc_id = grid_ids == mini_box_id
        padded_distance = 0.5 * minisize + padding
        absolute_rel_pos = np.abs(
            relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
        )
        mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

        return pos[mask], vel[mask], pid[mask]
    
def load_objects(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
    parallel: bool = True,
    name: str = "gas",
)-> Tuple[np.ndarray]:
    """Load particles from a mini box including all particles in adjacent boxes 
    up to the `padding` distance.

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    load_path : str
        Location from where to load the file
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5.

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, and PID
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    )

    # Load all adjacent boxes
    if not parallel:
        # Create empty lists (containers) to save the data from file for each ID
        pos, vel, pid, mass = ([] for _ in range(4))

        # Load all adjacent boxes
        for i, mini_box in enumerate(mini_box_ids):
            file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box}.hdf5'
            with h5.File(load_path + file_name, 'r') as hdf:
                pos.append(hdf[f'{name}/pos'][()])
                vel.append(hdf[f'{name}/vel'][()])
                pid.append(hdf[f'{name}/ID'][()])
                mass.append(hdf[f'{name}/Mass'][()])

        # Concatenate into a single array
        pos = np.concatenate(pos)
        vel = np.concatenate(vel)
        pid = np.concatenate(pid)
        mass = np.concatenate(mass)

        # Mask particles within a padding distance of the edge of the box in each
        # direction
        loc_id = grid_ids == mini_box_id
        padded_distance = 0.5 * minisize + padding
        absolute_rel_pos = np.abs(
            relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
        )
        mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

        return pos[mask], vel[mask], pid[mask], mass[mask]
    else:
        file_paths = [
            os.path.join(load_path, f'mini_boxes_nside_{boxes_per_side}', f'{bid}.hdf5')
            for bid in mini_box_ids
        ]

        results = Parallel(n_jobs=len(file_paths))(
            delayed(_load_one)(fp, name=name) for fp in file_paths
        )

        pos, vel, pid, mass = map(np.concatenate, zip(*results))

        # Mask particles within a padding distance of the edge of the box in each
        # direction

        loc_id = grid_ids == mini_box_id
        padded_distance = 0.5 * minisize + padding
        absolute_rel_pos = np.abs(
            relative_coordinates(pos, grid_pos[loc_id], boxsize, periodic=True)
        )
        mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

        return pos[mask], vel[mask], pid[mask], mass[mask]


def load_seeds(
    mini_box_id: int,
    boxsize: float,
    minisize: float,
    load_path: str,
    padding: float = 5.0,
) -> Tuple[np.ndarray]:
    """Load seeds from a mini box

    Parameters
    ----------
    mini_box_id : int
        Sub-box ID
    boxsize : float
        Size of simulation box
    minisize : float
        Size of mini box
    load_path : str
        Location from where to load the file
    padding : float
        Only particles up to this distance from the mini box edge are considered 
        for classification. Defaults to 5

    Returns
    -------
    Tuple[np.ndarray]
        Position, velocity, ID, R200b, M200b, Rs and a mask for seeds in the 
        minibox.
    """
    # Determine number of partitions per side
    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    # Generate the IDs and positions of the mini box grid
    grid_ids, grid_pos = generate_mini_box_grid(boxsize, minisize)

    # Get the adjacent mini box IDs
    mini_box_ids = get_adjacent_mini_box_ids(
        mini_box_id=mini_box_id,
        boxsize=boxsize,
        minisize=minisize
    )

    # Create empty lists (containers) to save the data from file for each ID
    pos, vel, hid, r200, m200, rs, mini_box_mask = ([] for _ in range(7))

    # Load all adjacent boxes
    for i, mini_box in enumerate(mini_box_ids):
        file_name = f'mini_boxes_nside_{boxes_per_side}/{mini_box}.hdf5'
        with h5.File(load_path + file_name, 'r') as hdf:
            pos.append(hdf['seed/pos'][()])
            vel.append(hdf['seed/vel'][()])
            hid.append(hdf['seed/ID'][()])
            r200.append(hdf['seed/R200b'][()])
            m200.append(hdf['seed/M200b'][()])
            rs.append(hdf['seed/Rs'][()])
            n_seeds = len(hdf['seed/ID'][()])
            if mini_box == mini_box_id:
                mini_box_mask.append(np.ones(n_seeds, dtype=bool))
            else:
                mini_box_mask.append(np.zeros(n_seeds, dtype=bool))

    # Concatenate into a single array
    pos = np.concatenate(pos)
    vel = np.concatenate(vel)
    hid = np.concatenate(hid)
    r200 = np.concatenate(r200)
    m200 = np.concatenate(m200)
    rs = np.concatenate(rs)
    mini_box_mask = np.concatenate(mini_box_mask)

    # Mask seeds within a padding distance of the edge of the box in each
    # direction
    loc_id = grid_ids == mini_box_id
    padded_distance = 0.5 * minisize + padding
    absolute_rel_pos = np.abs(relative_coordinates(
        pos, grid_pos[loc_id], boxsize, periodic=True,
    ))
    mask = np.prod(absolute_rel_pos <= padded_distance, axis=1, dtype=bool)

    m200 = m200[mask]
    r200 = r200[mask]
    pos = pos[mask]
    vel = vel[mask]
    hid = hid[mask]
    rs = rs[mask]
    mini_box_mask = mini_box_mask[mask]

    # Sort seeds by M200 (largest first)
    argorder = np.argsort(-m200)
    m200 = m200[argorder]
    r200 = r200[argorder]
    pos = pos[argorder]
    vel = vel[argorder]
    hid = hid[argorder]
    rs = rs[argorder]
    mini_box_mask = mini_box_mask[argorder]

    return (pos, vel, hid, r200, m200, rs, mini_box_mask)


# dealing with no downsampled particles

def process_subset(subset, source_path, save_path, boxsize, minisize, filename='flamingo_0077', part_name='part'):
    """Process one subset file"""
  
    filename = filename + f'.{subset}.hdf5'
    part = sw.load(join(source_path, filename))

    # basic quantities
    h = part.metadata.cosmology_raw['H0 [internal units]'][0]/100.0
    if part_name == 'part':
        part_id = part.dark_matter.particle_ids.value
        part_pos = part.dark_matter.coordinates.to(Mpc/h).value
        part_vel = part.dark_matter.velocities.to(km/s).value
        props = None
    elif part_name == 'gas':
        part_id = part.gas.particle_ids.value
        part_pos = part.gas.coordinates.to(Mpc/h).value
        part_vel = part.gas.velocities.to(km/s).value
        part_mass = part.gas.masses.to(Msun/h).value
        props = [part_mass,]
        labels = ("Mass",)
        dtypes = (np.float32,)
        props = (props, labels, dtypes)
    elif part_name == 'stars':
        part_id = part.stars.particle_ids.value
        part_pos = part.stars.coordinates.to(Mpc/h).value
        part_vel = part.stars.velocities.to(km/s).value
        part_mass = part.stars.masses.to(Msun/h).value
        props = [part_mass,]
        labels = ("Mass",)
        dtypes = (np.float32,)
        props = (props, labels, dtypes)

    boxes_per_side = np.int_(np.ceil(boxsize / minisize))

    chunk_size = part_id.shape[0] // (boxes_per_side**3)

    save_path_subset = join(save_path, f'subset/')

    split_box_into_mini_boxes(
        positions=part_pos,
        velocities=part_vel,
        uid=part_id,
        save_path=save_path_subset,
        boxsize=boxsize,
        minisize=minisize,
        chunksize=chunk_size,
        name=part_name,
        props=props,
        subset=subset,
    )


def get_all_dataset_paths(hf, prefix=''):
    """
    Recursively get all dataset paths in an HDF5 file or group.
    """
    paths = []
    for key in hf.keys():
        item = hf[key]
        path = f"{prefix}/{key}" if prefix else key
        if isinstance(item, h5.Group):
            paths.extend(get_all_dataset_paths(item, prefix=path))
        else:
            paths.append(path)
    return paths


def merge_one_minibox(mb_id, subset_dirs, output_dir):
    """Merge a single mini-box file from multiple subset directories into one."""
    out_path = os.path.join(output_dir, f"{mb_id}.hdf5")
    subset_files = [os.path.join(d, f"{mb_id}.hdf5") for d in subset_dirs 
                    if os.path.exists(os.path.join(d, f"{mb_id}.hdf5"))]

    if len(subset_files) == 0:
        return

    # Get dataset paths (supports multi-level groups)
    with h5.File(subset_files[0], 'r') as f0:
        all_keys = get_all_dataset_paths(f0)

    with h5.File(out_path, 'a') as hdf_out:
        for key in all_keys:
            data_list = []
            for file_i in subset_files:
                with h5.File(file_i, 'r') as hf:
                    data_list.append(hf[key][:])
            merged_data = np.concatenate(data_list, axis=0)

            grp_path = os.path.dirname(key)
            if grp_path != '':
                grp = hdf_out.require_group(grp_path)
            else:
                grp = hdf_out
            dset_name = os.path.basename(key)
            if dset_name in grp:
                del grp[dset_name]
            grp.create_dataset(dset_name, data=merged_data, compression="gzip", chunks=True, dtype=merged_data.dtype)


def merge_miniboxes_parallel(subset_dirs, output_dir, n_jobs=32):
    os.makedirs(output_dir, exist_ok=True)

    # Find all mini-box files in all subsets
    all_files = []
    for d in subset_dirs:
        all_files.extend(glob(os.path.join(d, "*.hdf5")))

    # Extract all file IDs (deduplicate and sort)
    box_ids = sorted({int(os.path.basename(f).split('.')[0]) for f in all_files})

    # Use joblib for parallel processing
    Parallel(n_jobs=n_jobs)(
        delayed(merge_one_minibox)(mb_id, subset_dirs, output_dir) 
        for mb_id in tqdm(box_ids, ncols=100, desc="Merging mini-boxes")
    )


def split_subsets_into_mini_boxes(num_subsets, source_path, save_path, boxsize, minisize, filename='flamingo_0077', part_name='part', n_threads=32):

    # parallel processing
    results = Parallel(n_jobs=n_threads, backend='loky')(
        delayed(process_subset)(i, source_path, save_path, boxsize=boxsize, minisize=minisize, filename=filename, part_name=part_name) for i in tqdm(range(num_subsets), ncols=100, desc="Processing subsets")
    )
    print("All subsets processed. Start to merge subsets...")
    # merge files

    save_path_subset = join(save_path, 'subset/')
    subset_dirs = [os.path.join(save_path_subset, f"subset_{i}") for i in range(num_subsets)]

    boxes_per_side = np.int_(np.ceil(boxsize / minisize))
    output_dir = save_path + f'mini_boxes_nside_{boxes_per_side}/'

    merge_miniboxes_parallel(subset_dirs, output_dir, n_jobs=n_threads)

    shutil.rmtree(save_path_subset)


if __name__ == "__main__":
    pass
