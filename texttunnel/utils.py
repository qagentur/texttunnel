import json
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

    The format is a JSON list of dictionaries, where each dictionary has an
    "id" key and a "text" key. The "id" key is an integer, and the "text" key
    is a string. This array of maps structure is easiest to parse by GPT models
    and handles edge cases like newlines in the text.

    Args:
        texts: A list of texts to format.

    Returns:
        A formatted string that can be used as a user message.
    """

    text_dicts = [
        {
            "id": i,
            "text": text,
        }
        for i, text in enumerate(texts)
    ]

    return json.dumps(text_dicts)


def format_texts_with_spaces(texts: List[str]) -> str:
    """
    Simple formatter that joins texts with spaces.
    """
    return " ".join(texts)


def binpack_texts_in_order(
    texts: List[str],
    max_tokens: int,
    formatter_function: Callable[[List[str]], str],
    max_texts: Optional[int] = None,
    encoding_name: str = "cl100k_base",
    long_text_handling: str = "error",
) -> List[List[str]]:
    """
    Binpacks a list of texts into a list of lists of texts, such that each list of texts
    has a total number of tokens less than or equal to max_tokens and each list of texts
    has a number of texts less than or equal to max_texts.

    The binpacking uses a naive greedy algorithm that maintains the order of the texts.

    Args:
        texts: The texts to binpack. Empty texts are accepted, counted as 0 tokens
            each and count against max_texts.
        formatter_function: A function that takes a list of texts and returns a single
            text. Defaults to None, which means that the texts are joined with spaces.
            This function is used to include the overhead of the formatter function in
            the binpacking. It is not used to format the output. Make sure to use
            the same formatter function when formatting the output for the model.
        max_tokens: The maximum number of tokens per list of texts. Leave some room for
            relative to the model's context size to account for the tokens in the
            system message, function call, and function return.
        max_texts: The maximum number of texts per list of texts. Defaults to None, which
            means that there is no limit on the number of texts per list of texts.
        encoding_name: The name of the encoding to use. Defaults to "cl100k_base".
        long_text_handling: How to handle texts that are longer than max_tokens. Defaults
            to "error", which means that an error is raised. Can also be set to
            "truncate", which means that the text is truncated to max_tokens.

    Returns:
        A list of lists of texts. The order of the texts is preserved.
    """

    if not max_texts:
        max_texts = len(texts)

    encoding = tiktoken.get_encoding(encoding_name)

    # Binpack the texts
    # Initialize the first bin
    bins = []
    current_bin = []
    current_bin_texts = 0

    for i, text in enumerate(texts):
        # Check if we need to start a new bin
        # Calculate how many tokens would be in the current bin if we added the text
        bin_tokens_with_new_text = len(
            encoding.encode(formatter_function(current_bin + [text]))
        )

        if bin_tokens_with_new_text > max_tokens or current_bin_texts == max_texts:
            # Start a new bin
            bins.append(current_bin)
            current_bin = []
            current_bin_texts = 0

            # Check if the text is too long to fit in a bin
            tokens_new_text_formatted = len(encoding.encode(formatter_function([text])))
            tokens_new_text_raw = len(encoding.encode(text))
            overhead = tokens_new_text_formatted - tokens_new_text_raw

            if overhead > max_tokens:
                raise ValueError(
                    f"""
                    The formatting function adds {overhead} overhead tokens,
                    which exceeds the maximum number of tokens ({max_tokens}) permitted.
                    """
                )

            if tokens_new_text_formatted > max_tokens:
                # The formatted text is too long to fit in a bin
                if long_text_handling == "error":
                    raise ValueError(
                        f"""
                        The text at index {i} has {bin_tokens_with_new_text} tokens, which
                        is greater than the maximum number of tokens ({max_tokens}).
                        Note that a formatting function added {overhead} tokens to the text.
                        """
                    )

                elif long_text_handling == "truncate":
                    text = truncate_text_by_tokens(
                        text=text,
                        max_tokens=max_tokens - overhead,
                        encoding=encoding,
                    )

                    assert (
                        len(encoding.encode(formatter_function([text]))) <= max_tokens
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
        current_bin_texts += 1

    # Add the last bin if it's not empty
    if current_bin:
        bins.append(current_bin)

    return bins