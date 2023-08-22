# Test fixtures shared across test files

import pytest
import json

import tiktoken

from texttunnel import chat, models


@pytest.fixture
def texts_fixture():
    return [
        "The first text.",
        "",  # empty string
        "The third text has non-ASCII characters: 你好世界",  # hello world in Chinese
        "The fourth text has a newline.\n",
    ]


@pytest.fixture
def texts_long_fixture():
    n_texts = 100
    min_length = 10
    max_length = 1000

    text_lengths = range(
        min_length, max_length + 10, 10  # range is not inclusive
    )  # 100 variations of text length

    j = 0

    # Cycle through texts lengths to create a list of texts
    texts = []
    for _ in range(n_texts):
        text_length = text_lengths[j]
        text = " ".join(["hello"] * text_length)  # Nirvana lyrics generator
        texts.append(text)
        if j < len(text_lengths) - 1:
            j += 1
        else:
            j = 0

    return texts


@pytest.fixture
def texts_nonascii_fixture():
    return [
        "Äpfel",  # apples in German
        "👋 🌍",
        "你好世界",  # hello world in Chinese
    ]


@pytest.fixture
def encoding_fixture():
    return tiktoken.get_encoding("cl100k_base")


@pytest.fixture
def response_fixture():
    return [
        {
            "model": "gpt-3.5-turbo",
            "max_tokens": 50,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": "How are you?",
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
            "id": "chatcmpl-7nQcrnnrqATiOktw8nY0AsbfGXqrn",
            "object": "chat.completion",
            "created": 1692014777,
            "model": "gpt-3.5-turbo",
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
                    },
                    "finish_reason": "stop",
                },
            ],
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        },
    ]


@pytest.fixture
def chat_fixture():
    return chat.Chat(
        messages=[
            chat.ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            chat.ChatMessage(
                role="user",
                content="Hello, world!",
            ),
        ]
    )


@pytest.fixture
def function_fixture():
    return {
        "name": "function_name",
        "parameters": {
            "type": "object",
            "properties": {"argument1": {"type": "string"}},
        },
    }


@pytest.fixture
def model_fixture():
    return models.Model(
        name="gpt-3.5-turbo",
        context_size=4096,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )


@pytest.fixture
def params_fixture():
    return models.Parameters(max_tokens=128)
