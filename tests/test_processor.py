from pathlib import Path
from texttunnel import processor


def test_prepare_output_filepath_tempfile():
    path = processor.prepare_output_filepath(None, keep_file=False)
    try:
        assert isinstance(path, Path)
        assert path.exists()
    finally:
        path.unlink()
