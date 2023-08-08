from texttunnel.chat import (
    ChatMessage,
    Chat,
    ChatCompletionRequest,
    Model,
    num_tokens_from_text,
    binpack_texts_in_order,
)
import pytest


@pytest.fixture
def chat_fixture():
    return Chat(
        messages=[
            ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            ChatMessage(
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
    return Model(
        name="gpt-3.5-turbo",
        context_size=4000,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )


@pytest.fixture
def input_texts():
    return [
        "The first text.",
        "",  # empty string
        "The third text has non-ASCII characters: 你好世界",  # hello world in Chinese
        "The fourth text has a newline.\n",
    ]


def test_num_tokens_from_text(input_texts):
    num_tokens = [num_tokens_from_text(text) for text in input_texts]
    assert num_tokens == [4, 0, 15, 7]


def test_binpack_texts_in_order(input_texts):
    bins = binpack_texts_in_order(input_texts, max_tokens=20)

    assert len(bins) == 2
    assert len(bins[0]) == 3


def test_chat(chat_fixture):
    chat = chat_fixture

    assert chat.messages[0].role == "system"


def test_chat_completion_request(model_fixture, chat_fixture, function_fixture):
    request = ChatCompletionRequest(
        model=model_fixture,
        chat=chat_fixture,
        function=function_fixture,
    )

    assert request.function_call == {"name": "function_name"}
    assert request.num_tokens_from_text() > 0
