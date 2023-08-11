import json
from pathlib import Path

import pytest

from texttunnel import processor


@pytest.fixture
def response_fixture():
    return [
        {
            "model": "gpt-3.5-turbo",
            "messages": [
                {
                    "role": "system",
                    "text": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "text": "How are you?",
                },
            ],
            "functions": [
                {
                    "name": "tell_feeling",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "feeling": {
                                "type": "string",
                                "enum": ["happy", "sad", "angry"],
                            }
                        },
                    },
                }
            ],
        },
        {
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "function_call": {
                            "name": "tell_feeling",
                            "arguments": json.dumps({"feeling": "happy"}),
                        },
                        "finish_reason": "stop",
                    },
                }
            ]
        },
    ]


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
