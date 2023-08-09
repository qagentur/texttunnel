from texttunnel import utils
import pytest
import tiktoken


@pytest.fixture
def texts_fixture():
    return [
        "The first text.",
        "",  # empty string
        "The third text has non-ASCII characters: 你好世界",  # hello world in Chinese
        "The fourth text has a newline.\n",
    ]


@pytest.fixture
def encoding_fixture():
    return tiktoken.get_encoding("cl100k_base")


def test_num_tokens_from_text(texts_fixture):
    num_tokens = [utils.num_tokens_from_text(text) for text in texts_fixture]
    assert num_tokens == [4, 0, 15, 7]


def test_binpack_texts_in_order(texts_fixture, encoding_fixture):
    max_tokens = 40
    text_bins = utils.binpack_texts_in_order(
        texts=texts_fixture,
        max_tokens=max_tokens,
        formatter_function=utils.format_texts_as_json,
    )

    tokens_in_bins = [
        len(encoding_fixture.encode(utils.format_texts_as_json(text_bin)))
        for text_bin in text_bins
    ]
    assert all([tokens <= max_tokens for tokens in tokens_in_bins])


def test_binpack_texts_in_order_long_text_error(texts_fixture):
    with pytest.raises(ValueError):
        utils.binpack_texts_in_order(
            texts=texts_fixture,
            max_tokens=5,
            formatter_function=utils.format_texts_as_json,
        )


def test_binpack_texts_in_order_truncation(texts_fixture, encoding_fixture):
    max_tokens = 25
    text_bins = utils.binpack_texts_in_order(
        texts=texts_fixture,
        max_tokens=max_tokens,
        formatter_function=utils.format_texts_as_json,
        long_text_handling="truncate",
    )

    tokens_in_bins = [
        len(encoding_fixture.encode(utils.format_texts_as_json(text_bin)))
        for text_bin in text_bins
    ]
    assert all([tokens <= max_tokens for tokens in tokens_in_bins])


def test_format_texts_as_json(texts_fixture):
    act = utils.format_texts_as_json(texts_fixture[:2])
    exp = '[{"id": 0, "text": "The first text."}, {"id": 1, "text": ""}]'

    assert act == exp


def truncate_text_by_tokens(encoding_fixture):
    text = "Hello, world!"
    truncated_text = utils.truncate_text_by_tokens(
        text, max_tokens=2, encoding=encoding_fixture
    )
    assert truncated_text == "Hello,"
