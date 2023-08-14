import json
from hashlib import sha256
from typing import Callable, List, Optional

import tiktoken


def num_tokens_from_text(text: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a string.

    Args:
        text: The text to count tokens in.
        encoding_name: The name of the token encoding to use. Defaults to "cl100k_base".

    Returns:
        The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(text))
    return num_tokens


def truncate_text_by_tokens(
    text: str,
    max_tokens: int,
    encoding: tiktoken.core.Encoding,
) -> str:
    """
    Truncates a text to a maximum number of tokens.

    Args:
        text: The text to truncate.
        max_tokens: The maximum number of tokens to truncate the text to.
        encoding: The encoding to use.

    Returns:
        The truncated text.
    """
    tokens = encoding.encode(text)
    truncated_tokens = tokens[:max_tokens]
    truncated_text = encoding.decode(truncated_tokens)

    return truncated_text


def format_texts_as_json(texts: List[str]) -> str:
    """
    Formats a list of texts into a single string to be used as a user message.
    Each text is assigned an ID, starting from 0. The returned JSON format
    helps the model distinguish between different texts, at the cost of
    increasing the number of tokens used.

    The token overhead for a single text that doesn't require escaping characters
    is 12 tokens. Escaping characters like quotes increases the overhead.

    The format is a JSON list of dictionaries, where each dictionary has an
    "id" key and a "text" key. The "id" key is an integer, and the "text" key
    is a string. This array of maps structure is easiest to parse by GPT models
    and handles edge cases like newlines in the text.

    Args:
        texts: A list of texts to format.

    Returns:
        A formatted string that can be used as a user message.
    """

    if not isinstance(texts, list):
        raise ValueError("texts must be a list.")

    text_dicts = [
        {
            "id": i,
            "text": text,
        }
        for i, text in enumerate(texts)
    ]

    # json.dumps escapes characters like quotes, which increases the token overhead
    # OpenAI models understand non-ascii characters, so we can use ensure_ascii=False
    # to avoid escaping characters.
    return json.dumps(text_dicts, ensure_ascii=False)


def format_texts_with_spaces(texts: List[str]) -> str:
    """
    Simple formatter that joins texts with spaces.
    """
    return " ".join(texts)


def binpack_texts_in_order(
    texts: List[str],
    formatter_function: Callable[[List[str]], str],
    max_tokens_per_bin: int,
    max_texts_per_bin: Optional[int] = None,
    encoding_name: str = "cl100k_base",
    long_text_handling: str = "error",
) -> List[List[str]]:
    """
    Binpacks a list of texts into a list of lists of texts, such that each list of texts
    has a total number of tokens less than or equal to max_tokens_per_bin and each list of texts
    has a number of texts less than or equal to max_texts_per_bin.

    The binpacking uses a naive greedy algorithm that maintains the order of the texts.

    Args:
        texts: List of texts to binpack. Empty texts are accepted, counted as 0 tokens
            each and count against max_texts_per_bin.
        formatter_function: A function that takes a list of texts and returns a single
            text. Defaults to None, which means that the texts are joined with spaces.
            This function is used to include the overhead of the formatter function in
            the binpacking. It is not used to format the output. Make sure to use
            the same formatter function when formatting the output for the model.
        max_tokens_per_bin: The maximum number of tokens per bin of formatted texts.
            Leave some room for relative to the model's context size to account for the tokens in the
            system message, function call, and function return.
        max_texts_per_bin: The maximum number of texts per list of texts. Defaults to None, which
            means that there is no limit on the number of texts per list of texts.
        encoding_name: The name of the encoding to use. Defaults to "cl100k_base".
        long_text_handling: How to handle texts that are longer than max_tokens_per_bin. Defaults
            to "error", which means that an error is raised. Can also be set to
            "truncate", which means that the text is truncated to max_tokens_per_bin.
            It is possible that more tokens are truncated than absolutely necessary
            due to overhead of the formatter function caused by escaping characters.

    Returns:
        A list of lists of texts. The order of the texts is preserved.
    """

    if not isinstance(texts, list):
        raise ValueError("texts must be a list.")

    if not max_texts_per_bin:
        max_texts_per_bin = len(texts)

    if max_texts_per_bin < 1:
        raise ValueError(
            f"max_texts_per_bin must be at least 1, but got {max_texts_per_bin}"
        )

    encoding = tiktoken.get_encoding(encoding_name)

    # Binpack the texts
    # Initialize the first bin
    bins = []
    current_bin = []

    for i, text in enumerate(texts):
        if len(current_bin) == max_texts_per_bin:
            # Start a new bin
            bins.append(current_bin)
            current_bin = []

        # Calculate how many tokens would be in the current bin if we added the text
        bin_tokens_with_new_text = len(
            encoding.encode(formatter_function(current_bin + [text]))
        )

        if bin_tokens_with_new_text > max_tokens_per_bin:  # doesn't fit
            if len(current_bin) > 0:
                # Start a new bin
                bins.append(current_bin)
                current_bin = []

            # Check if the text fits in a bin by itself
            tokens_text_with_formatting = len(
                encoding.encode(formatter_function([text]))
            )

            if tokens_text_with_formatting > max_tokens_per_bin:  # doesn't fit
                # Calculate the overhead of the formatter function
                tokens_text_raw = len(encoding.encode(text))
                overhead = tokens_text_with_formatting - tokens_text_raw

                if overhead > max_tokens_per_bin:
                    raise ValueError(
                        f"""
                        The formatting function adds {overhead} overhead tokens,
                        which exceeds the maximum number of tokens ({max_tokens_per_bin}) permitted.
                        """
                    )

                if bin_tokens_with_new_text > max_tokens_per_bin:
                    # The formatted text is too long to fit in a bin
                    if long_text_handling == "error":
                        raise ValueError(
                            f"""
                            The text at index {i} has {tokens_text_with_formatting} tokens, which
                            is greater than the maximum number of tokens ({max_tokens_per_bin}).
                            Note that a formatting function added {overhead} tokens to the text.
                            """
                        )

                    elif long_text_handling == "truncate":
                        # Truncate the text, accounting for overhead
                        # It's possible that more is truncated than necessary
                        # in case the overhead was caused by escaping characters
                        # in the truncated part of the text
                        text = truncate_text_by_tokens(
                            text=text,
                            max_tokens=max_tokens_per_bin - overhead,
                            encoding=encoding,
                        )

                        assert (
                            len(encoding.encode(formatter_function([text])))
                            <= max_tokens_per_bin
                        )

                    else:
                        raise ValueError(
                            f"""
                            Invalid value for long_text_handling: {long_text_handling}.
                            Must be one of "error" or "truncate".
                            """
                        )

        # Add to the current bin
        current_bin.append(text)

    # Add the last bin
    bins.append(current_bin)

    return bins


def hash_dict(d: dict) -> str:
    """
    Hashes a dictionary using sha256.
    """
    return sha256(json.dumps(d).encode("utf-8")).hexdigest()
