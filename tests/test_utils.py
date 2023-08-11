import pytest
import tiktoken
import itertools

from texttunnel import utils


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


def test_num_tokens_from_text(texts_fixture):
    num_tokens = [utils.num_tokens_from_text(text) for text in texts_fixture]
    assert num_tokens == [4, 0, 15, 7]


def test_binpack_texts_in_order(texts_fixture, encoding_fixture):
    max_tokens_per_bin = 40
    text_bins = utils.binpack_texts_in_order(
        texts=texts_fixture,
        max_tokens_per_bin=max_tokens_per_bin,
        formatter_function=utils.format_texts_as_json,
    )

    tokens_in_bins = [
        len(encoding_fixture.encode(utils.format_texts_as_json(text_bin)))
        for text_bin in text_bins
    ]
    assert all([tokens <= max_tokens_per_bin for tokens in tokens_in_bins])


def test_binpack_texts_in_order_overhead_too_long_error(texts_fixture):
    with pytest.raises(ValueError):
        utils.binpack_texts_in_order(
            texts=texts_fixture,
            max_tokens_per_bin=5,  # too small for overhead (12)
            formatter_function=utils.format_texts_as_json,
        )


def test_binpack_texts_in_order_text_too_long_error(texts_long_fixture):
    max_tokens_per_bin = 1000
    with pytest.raises(ValueError):  # Doesn't fit due to overhead
        utils.binpack_texts_in_order(
            texts=[texts_long_fixture[-1]],  # Last text has 1000 tokens
            max_tokens_per_bin=max_tokens_per_bin,
            formatter_function=utils.format_texts_as_json,
            long_text_handling="error",
        )


def test_binpack_texts_in_order_truncation(texts_fixture, encoding_fixture):
    max_tokens_per_bin = 25
    text_bins = utils.binpack_texts_in_order(
        texts=texts_fixture,
        max_tokens_per_bin=max_tokens_per_bin,
        formatter_function=utils.format_texts_as_json,
        long_text_handling="truncate",
    )

    tokens_in_bins = [
        len(encoding_fixture.encode(utils.format_texts_as_json(text_bin)))
        for text_bin in text_bins
    ]
    assert all([tokens <= max_tokens_per_bin for tokens in tokens_in_bins])


def test_binpack_texts_in_order_long_texts(texts_long_fixture, encoding_fixture):
    max_tokens_per_bin = 1013  # exactly fits the longest text including overhead
    text_bins = utils.binpack_texts_in_order(
        texts=texts_long_fixture,
        max_tokens_per_bin=max_tokens_per_bin,
        formatter_function=utils.format_texts_as_json,
        long_text_handling="error",
    )

    # All texts should be in a bin
    flattened_bins = list(itertools.chain.from_iterable(text_bins))
    assert len(flattened_bins) == len(texts_long_fixture)

    tokens_in_bins = [
        len(encoding_fixture.encode(utils.format_texts_as_json(text_bin)))
        for text_bin in text_bins
    ]
    assert all([tokens <= max_tokens_per_bin for tokens in tokens_in_bins])


def test_binpack_texts_in_order_max_texts_per_bin(texts_long_fixture):
    max_tokens_per_bin = 10000  # very large
    max_texts_per_bin = 3

    text_bins = utils.binpack_texts_in_order(
        texts=texts_long_fixture,
        max_tokens_per_bin=max_tokens_per_bin,
        max_texts_per_bin=max_texts_per_bin,
        formatter_function=utils.format_texts_as_json,
    )

    assert max([len(text_bin) for text_bin in text_bins]) == max_texts_per_bin


def test_format_texts_as_json(texts_fixture):
    act = utils.format_texts_as_json(texts_fixture[:2])
    exp = '[{"id": 0, "text": "The first text."}, {"id": 1, "text": ""}]'

    assert act == exp


def test_format_texts_as_json_keeps_non_ascii_characters(texts_nonascii_fixture):
    act = utils.format_texts_as_json(texts_nonascii_fixture)
    exp = '[{"id": 0, "text": "Äpfel"}, {"id": 1, "text": "👋 🌍"}, {"id": 2, "text": "你好世界"}]'

    assert act == exp


def test_truncate_text_by_tokens(encoding_fixture):
    text = "Hello, world!"
    truncated_text = utils.truncate_text_by_tokens(
        text, max_tokens=2, encoding=encoding_fixture
    )
    assert truncated_text == "Hello,"
