import numpy
import pytest

from oasis import catalogue


def test_characteristic_density():
    r200 = 1.
    rs = 1.

    # Test zero division
    with pytest.raises(ZeroDivisionError):
        catalogue.characteristic_density(r200=r200, rs=0.)
    
    with pytest.raises(ZeroDivisionError):
        catalogue.characteristic_density(r200=0., rs=rs)

    with pytest.raises(ZeroDivisionError):
        catalogue.characteristic_density(r200=numpy.arange(3), rs=numpy.arange(3))
    
    delta = catalogue.characteristic_density(r200=r200, rs=rs)
    assert delta == pytest.approx(200. / 3. / (numpy.log(2) - 0.5) )



def test_rho_nfw_roots():
    
    r200 = 1.
    rs = 1.
    delta = catalogue.characteristic_density(r200=r200, rs=rs)
    x = numpy.linspace(0, 5., 1_000)
    froot = catalogue.rho_nfw_roots(x=x, delta1=delta, rs1=rs, delta2=delta, 
                                    rs2=rs, r12=2.5)

    # assert all(froot == 0.)