from texttunnel import chat
import pytest


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
    return chat.Model(
        name="gpt-3.5-turbo",
        context_size=4096,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )


def test_is_valid_function_def(function_fixture):
    assert chat.is_valid_function_def(function_fixture)

    bad_function = function_fixture.copy()
    del bad_function["name"]

    assert not chat.is_valid_function_def(bad_function)


def test_chat(chat_fixture):
    chat = chat_fixture

    assert chat.messages[0].role == "system"


def test_chat_completion_request(model_fixture, chat_fixture, function_fixture):
    request = chat.ChatCompletionRequest(
        model=model_fixture,
        chat=chat_fixture,
        function=function_fixture,
    )

    assert request.function_call == {"name": "function_name"}
    assert request.count_tokens() > 0


def test_chat_completion_request_context_size_check(chat_fixture, function_fixture):
    tiny_model = chat.Model(
        name="tiny-model",
        context_size=1,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )

    with pytest.raises(ValueError):
        chat.ChatCompletionRequest(
            model=tiny_model,
            chat=chat_fixture,
            function=function_fixture,
        )
