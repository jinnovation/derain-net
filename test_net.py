import numpy as np
import net
import pytest

class TestConvolve():
    @pytest.mark.parametrize("p_in,kernels,expected", [
        (
            np.array([
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]],
            ]),
            [
                np.array([
                    [[1],[2]],
                    [[3],[4]],
                ]),
            ],
            np.array([
                [[37], [47]],
                [[67], [77]],
            ]),
        ),
        (
            np.array([
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]],
            ]),
            [
                np.array([
                    [[1],[2]],
                    [[3],[4]],
                    [[5],[6]],
                ]),
            ],
            np.array([
                [[120],[141]],
            ]),
        ),
        (
            np.array([
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]],
            ]),
            [
                np.array([[[1]]]),
            ],
            np.array([
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]],
            ]),
        ),
        (
            np.array([
                [[1,1,1], [2,2,2], [3,3,3]],
                [[4,4,4], [5,5,5], [6,6,6]],
                [[7,7,7], [8,8,8], [9,9,9]],
            ]),
            [
                np.array([
                    [[1,1,1], [2,2,2]],
                    [[3,3,3], [4,4,4]],
                ]),
            ],
            np.array([
                [[111], [141]],
                [[201], [231]],
            ]),
        ),
        (
            np.array([
                [[1],[2],[3]],
                [[4],[5],[6]],
                [[7],[8],[9]],
            ]),
            [
                np.array([[[1]]]),
                np.array([[[2]]]),
            ],
            np.array([
                [[1,2],[2,4],[3,6]],
                [[4,8],[5,10],[6,12]],
                [[7,14],[8,16],[9,18]],
            ]),
        ),
    ])
    def test_output_values(self, p_in, kernels, expected):
        out = net.convolve(p_in, *kernels)
        np.testing.assert_allclose(out, expected)

    @pytest.mark.parametrize("d_in,ds_k", [
        ((2,2,1), [(3,3,1)]),
        ((2,2,1), [(0,0,1)]),
        ((2,2,1), [(1,1,2)]),
        ((2,2,1), [(1,1,1),(1,1,2)]),
    ])
    def test_raises_dimension_error_2d(self, d_in, ds_k):
        patch_in, kernels = np.empty(d_in), [np.empty(d_k) for d_k in ds_k]
        pytest.raises(net.DimensionException, net.convolve, patch_in, *kernels)
