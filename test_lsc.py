""" Test module for least square classification.

@authors: Valentin
"""
import pytest
from generate_data import generate_gaussian_data, generate_K_gaussians


def test_generate_gaussian_data():
    """ Test generate gaussian data return values. """
    data_generated = generate_gaussian_data(15, (2, 2), (1, 1))
    assert len(data_generated) == 15
    assert len(data_generated[0]) == 2


def test_generate_K_gaussians():
    """ Test generate K gaussian return values. """
    # Check wrong arguments passed
    n = 20
    mus = (1,)
    sigmas = (0.1,)
    with pytest.raises(TypeError) as excinfo:
        generate_K_gaussians(n, mus, sigmas)
    assert 'You need to input a list of mus and sigmas' in str(excinfo.value)

    # Check different sizes of mus and sigmas
    n = 20
    mus = [(1,)]
    sigmas = [(0.1,), (0.2)]
    with pytest.raises(AssertionError) as excinfo:
        generate_K_gaussians(n, mus, sigmas)
    assert 'You need as much mean as variance' in str(excinfo.value)

    # check differtent dimentions
    n = 20
    mus = [(1,)]
    sigmas = [(0.1, 0.2)]
    with pytest.raises(AssertionError) as excinfo:
        generate_K_gaussians(n, mus, sigmas)
    assert 'Dimension of mean and variance must be the same'\
        in str(excinfo.value)

    # check Normal functionnment 1 gaussian
    n = 20
    mus = [(1, 2)]
    sigmas = [(0.1, 0.2)]
    out = generate_K_gaussians(n, mus, sigmas)
    assert len(out) == 1
    assert len(out[0]) == 20
    assert len(out[0][0]) == 2

    # check Normal functionnment 3 gaussian
    n = 20
    mus = [(1, 2), (0, 0), (3, 3)]
    sigmas = [(0.1, 0.1), (0.1, 0.1), (0.1, 0.1)]
    out = generate_K_gaussians(n, mus, sigmas)
    assert len(out) == 3
    assert len(out[0]) == 20
    assert len(out[1]) == 20
    assert len(out[2]) == 20
    assert len(out[0][0]) == 2
