from texttunnel.chat import ChatMessage, Chat, ChatCompletionRequest, Model
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
        model="gpt-3.5-turbo",
        context_size=4000,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )


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
