import numpy as np
import net
import pytest

class TestNet():
    @pytest.mark.parametrize("p_in,kernel,expected", [
        (
            np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ]),
            np.array([
                [1,2],
                [3,4],
            ]),
            np.array([
                [37, 47],
                [67, 77],
            ]),
        ),
        (
            np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ]),
            np.array([
                [1,2],
                [3,4],
                [5,6],
            ]),
            np.array([
                [120,141],
            ]),
        ),
        (
            np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ]),
            np.array([[1]]),
            np.array([
                [1,2,3],
                [4,5,6],
                [7,8,9],
            ]),
        )
    ])
    def test_convolve_output_values(self, p_in, kernel, expected):
        np.testing.assert_allclose(net.convolve(p_in,kernel), expected)

    @pytest.mark.parametrize("d_in,d_k", [
        ((2,2), (3,3)),
        ((2,2), (0,0)),
    ])
    def test_convolve_raises_dimension_error_2d(self, d_in, d_k):
        patch_in, kernel = np.empty(d_in), np.empty(d_k)
        pytest.raises(net.DimensionException, net.convolve, patch_in, kernel)
            

