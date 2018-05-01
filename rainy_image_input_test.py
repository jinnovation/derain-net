from unittest import mock

def test__get_indices():
    with mock.patch("glob.glob") as MockGlob, \
        mock.patch("tensorflow.gfile.Exists") as MockExists:
        MockGlob.return_value = ["../foo/ground truth/1.jpg", "../foo/ground truth/2.jpg"]
        MockExists.return_value = True
        assert _get_indices("../foo") == ["1", "2"]
