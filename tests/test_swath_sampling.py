import numpy as np
import pytest
import datetime
from odysim.swath_sampling import OdyseaSwath
import warnings


@pytest.fixture
def first_orbit():
    # gets first orbit from OdyseaSwath
    orbits = OdyseaSwath()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, message="Converting non-nanosecond precision")
        o = next(orbits.getOrbits(datetime.datetime(2020, 1, 20), datetime.datetime(2020, 1, 21)))
    return o


@pytest.fixture
def expected_lat_lon_time():
    # get expected first orbit data
    with np.load('./tests/data/expected_swath_lat_lon_time.npz', 'r') as data:
        expected_lat = data['lat']
        expected_lon = data['lon']
        expected_time = data['sample_time']
    return expected_lat, expected_lon, expected_time


@pytest.fixture
def expected_azimuth_encoder_bearing():
    # get expected first orbit data
    with np.load('./tests/data/expected_swath_encoder_azimuth_bearing.npz', 'r') as data:
        expected_encoder_fore = data['encoder_fore']
        expected_encoder_aft = data['encoder_aft']
        expected_azimuth_fore = data['azimuth_fore']
        expected_azimuth_aft = data['azimuth_aft']
        expected_bearing = data['bearing']
    return expected_encoder_fore, expected_encoder_aft, expected_azimuth_fore, expected_azimuth_aft, expected_bearing


def test_swath_llt(first_orbit, expected_lat_lon_time):
    expected_lat, expected_lon, expected_time = expected_lat_lon_time
    o = first_orbit
    actual_lat = o['lat'].values
    actual_lon = o['lon'].values
    actual_time = o['sample_time'].values

    assert np.allclose(actual_lat, expected_lat, atol=1e-6)
    assert np.allclose(actual_lon, expected_lon, atol=1e-6)
    assert np.array_equal(actual_time, expected_time)


def test_swath_azimuth_encoder_bearing(first_orbit, expected_azimuth_encoder_bearing):
    expected_encoder_fore, expected_encoder_aft, expected_azimuth_fore, expected_azimuth_aft, expected_bearing = expected_azimuth_encoder_bearing
    o = first_orbit
    actual_encoder_fore = o['encoder_fore'].values
    actual_encoder_aft = o['encoder_aft'].values
    actual_azimuth_fore = o['azimuth_fore'].values
    actual_azimuth_aft = o['azimuth_aft'].values
    actual_bearing = o['bearing'].values

    assert np.allclose(actual_encoder_fore, expected_encoder_fore, atol=1e-6)
    assert np.allclose(actual_encoder_aft, expected_encoder_aft, atol=1e-6)
    assert np.allclose(actual_azimuth_fore, expected_azimuth_fore, atol=1e-6)
    assert np.allclose(actual_azimuth_aft, expected_azimuth_aft, atol=1e-6)
    assert np.allclose(actual_bearing, expected_bearing, atol=1e-6)
