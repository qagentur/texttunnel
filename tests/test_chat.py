from texttunnel import chat
import pytest
import tiktoken


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
        context_size=4000,
        input_token_price_per_1k=0.002,
        output_token_price_per_1k=0.004,
        tokens_per_minute=90000,
        requests_per_minute=3500,
    )


@pytest.fixture
def texts_fixture():
    return [
        "The first text.",
        "",  # empty string
        "The third text has non-ASCII characters: 你好世界",  # hello world in Chinese
        "The fourth text has a newline.\n",
    ]


def test_num_tokens_from_text(texts_fixture):
    num_tokens = [chat.num_tokens_from_text(text) for text in texts_fixture]
    assert num_tokens == [4, 0, 15, 7]


def test_binpack_texts_in_order(texts_fixture):
    bins = chat.binpack_texts_in_order(texts_fixture, max_tokens=20)

    assert len(bins) == 2
    assert len(bins[0]) == 3


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


def test_format_texts_as_json(texts_fixture):
    act = chat.format_texts_as_json(texts_fixture[:2])
    exp = '[{"id": 0, "text": "The first text."}, {"id": 1, "text": ""}]'

    assert act == exp


def test_build_binpacked_requests(
    model_fixture,
    function_fixture,
    texts_fixture,
):
    requests = chat.build_binpacked_requests(
        system_message="You are a helpful assistant.",
        model=model_fixture,
        function=function_fixture,
        texts=texts_fixture,
    )

    assert len(requests) == 1


def test_get_formatter_overhead():
    overhead = chat.get_formatter_overhead(
        formatter_function=chat.format_texts_as_json,
        encoding=tiktoken.get_encoding("cl100k_base"),
    )
    assert overhead > 0
