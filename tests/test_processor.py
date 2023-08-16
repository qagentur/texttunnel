from pathlib import Path
from texttunnel import processor


def test_prepare_output_filepath_tempfile():
    path = processor.prepare_output_filepath(None, keep_file=False)
    try:
        assert isinstance(path, Path)
        assert path.exists()
    finally:
        path.unlink()


def test_parse_response(response_fixture):
    act = processor.parse_response(response_fixture)
    exp = {"feeling": "happy"}

    assert act == exp
