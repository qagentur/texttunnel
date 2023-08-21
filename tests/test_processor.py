from pathlib import Path
from texttunnel import processor


def test_prepare_output_filepath_tempfile():
    path = processor.prepare_output_filepath(None, keep_file=False)
    try:
        assert isinstance(path, Path)
        assert path.exists()
    finally:
        path.unlink()


def test_is_valid_response(response_fixture):
    assert processor.is_valid_response(response_fixture, print_errors=True)


def test_is_valid_response_fails_on_invalid_response(response_fixture):
    invalid_response = response_fixture.copy()
    del invalid_response[1]["usage"]
    assert not processor.is_valid_response(invalid_response)


def test_parse_response(response_fixture):
    act = processor.parse_arguments(response_fixture)
    exp = {"feeling": "happy"}

    assert act == exp


def test_parse_token_usage(response_fixture):
    act = processor.parse_token_usage(response_fixture)
    exp = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    assert act == exp


def test_usage_to_cost(response_fixture, model_fixture):
    usage = processor.parse_token_usage(response_fixture)
    cost = processor.usage_to_cost(usage, model_fixture)

    assert cost > 0
