import pytest
import numpy as np
from src.models.vgm import ScalableVGM

def test_vgm_fit(vgm, sample_data):
    """Test model fitting with known modes"""
    vgm.fit(sample_data)
    
    # Check if means were identified correctly
    means = sorted(vgm.means)
    assert len(means) == 3
    assert abs(means[0] - 10) < 2
    assert abs(means[1] - 20) < 2
    assert abs(means[2] - 30) < 2

def test_vgm_transform(vgm, sample_data):
    """Test data transformation"""
    vgm.fit(sample_data)
    transformed = vgm.transform_batch(sample_data)
    
    # Check shape and normalization
    assert transformed.shape == (len(sample_data), 1)
    assert abs(np.mean(transformed)) < 0.5
    assert abs(np.std(transformed) - 1) < 0.5

def test_vgm_inverse_transform(vgm, sample_data):
    """Test inverse transformation"""
    vgm.fit(sample_data)
    transformed = vgm.transform_batch(sample_data)
    mode_indicators = np.zeros(len(transformed), dtype=int)
    reconstructed = vgm.inverse_transform_batch(transformed, mode_indicators)
    
    assert reconstructed.shape == (len(sample_data), 1)

def test_vgm_error_handling(vgm):
    """Test error handling"""
    with pytest.raises(Exception):
        vgm.fit(None)
    
    with pytest.raises(Exception):
        vgm.transform_batch(np.array([1, 2, 3]))  # Unfitted model

@pytest.mark.parametrize("batch_size", [1000, 10000, 100000])
def test_vgm_large_batch(vgm, batch_size):
    """Test processing of different batch sizes"""
    large_batch = np.random.normal(size=batch_size)
    vgm.fit(large_batch)
    transformed = vgm.transform_batch(large_batch)
    assert len(transformed) == batch_size

def test_vgm_component_validation(vgm, sample_data):
    """Test component validation"""
    vgm.fit(sample_data)
    assert vgm.valid_component_indicator is not None
    assert np.any(vgm.valid_component_indicator) 