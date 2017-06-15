import numpy as np
import net
import pytest

class TestNet():
    def test_convolve_output_values(self):
        p_in = np.matrix([
            [1,2,3],
            [4,5,6],
            [7,8,9],
        ])
        kernel = np.matrix([
            [1,2],
            [3,4],
        ])

        out = net.convolve(p_in, kernel)

        assert np.matrix([
            [1*1+2*2+3*4+4*5, 1*2+2*3+3*5+4*6],
            [1*4+2*5+3*7+4*8, 1*5+2*6+3*8+4*9],
        ]) == out

    def test_convolve_raises_dimension_error(self):
        for p_in, k in [
            (
                np.matrix([
                    [1,2],
                    [3,4],
                ]),
                np.matrix([
                    [1,2,3],
                    [4,5,6],
                    [7,8,9],
                ]),
            ),
        ]:
            pytest.raises(net.DimensionException, net.convolve, p_in, k)
            

