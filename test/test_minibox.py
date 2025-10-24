import os

import numpy
import pytest

from oasis import coordinates, minibox, common

l_box = 100.
l_mb = 20.


def test_generate_mini_box_grid():
    """Check if `generate_mini_box_grid` creates a regular grid."""
    ids, centres = minibox.generate_mini_box_grid(boxsize=l_box, minisize=l_mb)

    assert len(ids) == len(centres)  # Length of arrays is the same
    # Number of elements is (l_box/l_mb)**3
    assert len(ids) == numpy.int_(numpy.ceil(l_box / l_mb))**3
    # First position is shifted by l_mb/2
    assert all(centres[0] == numpy.full(3, 0.5*l_mb))


def test_get_mini_box_id():
    """Check if `get_mini_box_id` generates the right IDs for each particle."""
    _, centres = minibox.generate_mini_box_grid(boxsize=l_box, minisize=l_mb)
    n_mb = numpy.int_(numpy.ceil(l_box / l_mb))**3

    # Partition box and retrive subbox ID for each particle. Only one particle
    # per subbox.
    box_ids = minibox.get_mini_box_id(x=centres, boxsize=l_box, minisize=l_mb)

    assert box_ids[0] == 0  # First particle is in the first box with ID = 0
    # Last particle is in the last box with ID = 999
    assert box_ids[-1] == n_mb - 1
    assert len(numpy.unique(box_ids)) == n_mb  # One particle per subbox

    box_ids = minibox.get_mini_box_id(x=centres[0], boxsize=l_box, minisize=l_mb)
    assert box_ids == 0


def test_get_adjacent_mini_box_ids():
    """Check if the number of adjacent miniboxes is in fact 27."""

    adj_ids = minibox.get_adjacent_mini_box_ids(
        mini_box_id=0,
        boxsize=l_box,
        minisize=l_mb,
    )

    adj_ids_0 = [0, 1, 4, 5, 6, 9, 20, 21, 24, 25, 26, 29, 30, 31, 34, 45, 46,
                 49, 100, 101, 104, 105, 106, 109, 120, 121, 124]

    assert len(adj_ids) == 27
    assert all([i in adj_ids for i in adj_ids_0])


def test_generate_mini_box_ids():
    """Sort items into miniboxes according to their positions."""
    n_samples = 1000
    chunk_size = 10
    seed = 1234
    pos = coordinates.gen_data_pos_random(l_box, n_samples, seed)

    ids = minibox.generate_mini_box_ids(pos, l_box, l_mb, chunk_size)

    assert min(ids) == 0
    assert max(ids) <= numpy.int_(numpy.ceil(l_box / l_mb))**3
    assert len(ids) == n_samples


def test_get_chunks():
    ones = numpy.ones(100, dtype=numpy.uint8)
    with pytest.raises(IndexError):
        minibox.get_chunks(ids=ones, chunksize=1)

    nums = numpy.repeat(numpy.arange(10, dtype=numpy.uint8), 10)
    chunks = minibox.get_chunks(ids=nums, chunksize=15)
    assert len(chunks) == 11



def test_split_into_mini_boxes():
    # Create sinthetic data
    l_box = 500.
    l_mb = 100.
    n_points = numpy.int_(numpy.ceil(l_box / l_mb))**3
    pos = coordinates.gen_data_pos_regular(l_box, l_mb)
    vel = numpy.random.uniform(-1, 1, n_points)
    # Offset PIDs to avoid confusion with mini box ID.
    pid = numpy.arange(2000, 2000 + n_points)

    temp_dir = os.getcwd() + '/temp/'
    # common.mkdir(temp_dir)
    # minibox.split_box_into_mini_boxes(pos, vel, pid, temp_dir, l_box, l_mb, 
    #                                   chunksize=2*n_points)
    # os.removedirs(temp_dir)


def test_load_particles():
    ...


def test_load_seeds():
    ...

###
