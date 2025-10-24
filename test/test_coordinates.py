import numpy
import pytest

from oasis import coordinates


def test_cartesian_product():
    # List [0, 1, 2]
    n_points = 3
    points = numpy.arange(n_points)
    points_float = numpy.linspace(0, n_points-1, n_points)

    # Repeat the list twice and thrice
    arrs_2 = 2 * [points]
    arrs_3 = 3 * [points]

    cart_prod_1 = coordinates.cartesian_product([points])
    cart_prod_2 = coordinates.cartesian_product(arrs_2)
    cart_prod_3 = coordinates.cartesian_product(arrs_3)
    cart_prod_float = coordinates.cartesian_product([points, points_float])

    # Cardinality of Nx...xN = N^n
    assert len(cart_prod_1) == n_points
    assert len(cart_prod_2) == n_points*n_points
    assert len(cart_prod_3) == n_points*n_points*n_points
    # Each element has shape (n,)
    assert cart_prod_1[0].shape == (1,)
    assert cart_prod_2[0].shape == (2,)
    assert cart_prod_3[0].shape == (3,)
    # Check dtypes
    assert type(cart_prod_1[0][0]) == numpy.int64
    assert type(cart_prod_float[0][0]) == numpy.float64


def test_gen_data_pos_regular():
    """Check if `gen_data_pos_regular` creates a regular grid."""
    l_box = 100.
    l_mb = 20.

    pos = coordinates.gen_data_pos_regular(l_box, l_mb)

    # Number of elements is (l_box/l_mb)**3
    assert len(pos) == numpy.int_(numpy.ceil(l_box / l_mb))**3
    # First position is shifted by l_mb/2
    assert all(pos[0] == numpy.full(3, 0.5*l_mb))


def test_gen_data_pos_random():
    """Check if `gen_data_pos_random` generates the right number of samples and
    within the box."""
    l_box = 100.
    n_samples = 1000
    seed = 1234

    pos = coordinates.gen_data_pos_random(l_box, n_samples, seed)

    assert pos.shape == (n_samples, 3)
    assert numpy.max(pos) <= l_box
    assert numpy.min(pos) >= 0


def test_relative_coordinates():
    """Check if `relative_coordinates` gets the periodic boundary condition 
    right."""
    l_box = 100.
    x0 = 99.5
    x1 = 4.

    assert coordinates.relative_coordinates(x1, x0, l_box, periodic=False) < 0
    assert coordinates.relative_coordinates(x1, x0, l_box) < x0 - x1

    assert coordinates.relative_coordinates(numpy.array([x0, x1]), x0, 
                                            l_box).shape == (2, )
    with pytest.raises(TypeError):
        coordinates.relative_coordinates([x0, x1], x1, l_box)
    with pytest.raises(TypeError):
        coordinates.relative_coordinates((x0, x1), x1, l_box)


def test_get_vr_vt_from_coordinates():

    x = numpy.array([1., 1., 1.])
    v = numpy.array([1., 1., -1.])

    with pytest.raises(ValueError):
        coordinates.velocity_components(x, v)

    x = numpy.array([x])
    v = numpy.array([v])
    vr, vt, v2 = coordinates.velocity_components(x, v)

    vr_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) * \
        (numpy.cos(numpy.pi/4.) + numpy.sin(numpy.pi/4.)) - 1./numpy.sqrt(3.)
    vt_expected = numpy.sin(numpy.arccos(1./numpy.sqrt(3.))) + 1./numpy.sqrt(3.) * \
        (numpy.cos(numpy.pi/4.) + numpy.sin(numpy.pi/4.))

    assert vr[0] == pytest.approx(vr_expected)
    assert vt[0] == pytest.approx(vt_expected)
    assert v2[0] == 3.


###
