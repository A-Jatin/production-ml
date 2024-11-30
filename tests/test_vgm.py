import pytest
import numpy as np
from src.models.vgm import ScalableVGM

@pytest.mark.parametrize("data_shape", [(100, 10), (200, 20)])
def test_scalable_vgm(data_shape):
    data = np.random.rand(*data_shape)
    vgm = ScalableVGM()
    vgm.fit(data)
    assert vgm.means is not None
    assert vgm.stds is not None
