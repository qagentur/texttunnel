import pytest

from texttunnel import chat, models


def test_chat_add_message(chat_fixture):
    chat_fixture.add_message(message=chat.ChatMessage(role="user", content="Hi!"))
    assert len(chat_fixture.messages) == 3
    assert chat_fixture.messages[2].content == "Hi!"


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
        params=models.Parameters(max_tokens=128, temperature=0.5),
    )

    assert request.function_call == {"name": "function_name"}
    assert request.count_total_tokens() > 0
    assert request.estimate_cost_usd() > 0
    assert isinstance(request.to_dict(), dict)
    assert request.to_dict()["temperature"] == 0.5


def test_chat_completion_request_context_size_exceeded(
    model_fixture, chat_fixture, function_fixture
):
    with pytest.raises(ValueError):
        chat.ChatCompletionRequest(
            model=model_fixture,
            chat=chat_fixture,
            function=function_fixture,
            params=models.Parameters(max_tokens=4080),  # doesn't fit
        )


def test_build_binpacked_requests_default_settings(
    model_fixture,
    function_fixture,
    texts_long_fixture,
):
    requests = chat.build_binpacked_requests(
        system_message="You are a helpful assistant.",
        model=model_fixture,
        function=function_fixture,
        texts=texts_long_fixture,
    )

    assert all([r.count_total_tokens() <= model_fixture.context_size for r in requests])


def test_build_binpacked_requests_max_texts_per_request(
    model_fixture,
    function_fixture,
    texts_fixture,
):
    requests = chat.build_binpacked_requests(
        system_message="You are a helpful assistant.",
        model=model_fixture,
        function=function_fixture,
        texts=texts_fixture,
        max_texts_per_request=2,
    )

    assert len(requests) == 2


def test_build_requests(
    model_fixture,
    function_fixture,
    texts_fixture,
):
    requests = chat.build_requests(
        system_message="You are a helpful assistant.",
        model=model_fixture,
        function=function_fixture,
        texts=texts_fixture,
    )

    assert len(requests) == len(texts_fixture)


def test_chat_completion_request_context_size_check(chat_fixture, function_fixture):
    tiny_model = chat.Model(
        name="gpt-3.5-turbo",
        context_size=1,  # only for testing, real context size is 4096
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
