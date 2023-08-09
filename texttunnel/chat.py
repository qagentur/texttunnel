from typing import Any, Callable, Dict, List, Optional
import tiktoken
import json

from texttunnel.models import Model


FunctionDef = Dict[str, str]


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


class ChatMessage:
    """
    A chat message, to be used in a chat.

    Args:
        role: The role of the message. Must be one of "system", "user", or "assistant".
        content: The content of the message.
    """

    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(self, role: str, content: str):
        if role not in self.VALID_ROLES:
            raise ValueError(f"Invalid role {role}. Must be one of {self.VALID_ROLES}.")

        self.role = role
        self.content = content

    def to_dict(self) -> Dict[str, str]:
        """
        Returns a dict representation of the message.
        """
        return {"role": self.role, "content": self.content}

    def count_tokens(self) -> int:
        """
        Returns the number of tokens in the message.
        """
        return num_tokens_from_text(self.content)


class Chat:
    """
    A chat. Used to prompt a model for a response.
    The first message must be from the system, and the last message must be from the user.

    Args:
        messages: A list of ChatMessage objects.
    """

    def __init__(self, messages: List[ChatMessage]):
        if len(messages) < 2:
            raise ValueError("A chat must have at least two messages.")

        if messages[0].role != "system":
            raise ValueError("The first message in a chat must be from the system.")

        if messages[-1].role != "user":
            raise ValueError("The last message in a chat must be from the user.")

        self.messages = messages

    def __len__(self) -> int:
        """
        Returns the number of messages in the chat.
        """
        return len(self.messages)

    def add_message(self, message: ChatMessage) -> None:
        """
        Adds a message to the end of the chat.

        Args:
            message: The message to add.
        """
        self.messages.append(message)

    def to_list(self) -> List[Dict[str, str]]:
        """
        Returns a list of dictionaries representing the chat messages.
        This is the format expected by the OpenAI API.
        """
        return [message.to_dict() for message in self.messages]

    def count_tokens(self) -> int:
        """
        Returns the number of tokens in all of the messages in the chat.
        """
        return sum(message.count_tokens() for message in self.messages)


class ChatCompletionRequest:
    """
    Defines a request for a chat completion.

    Args:
        chat: The chat to which the assistant should respond with a function call.
        model: The name of the OpenAI ChatCompletion model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
        model_params: Additional keyword arguments to pass to the OpenAI API. See
            https://platform.openai.com/docs/api-reference/completions/create
    """

    def __init__(
        self,
        chat: Chat,
        model: Model,
        function: FunctionDef,
        model_params: Optional[Dict[str, Any]] = None,
    ):
        self.chat = chat
        self.model = model

        self.function = function

        # Force the model to use a function call
        self.functions = [function]
        self.function_call = {"name": function["name"]}

        self.model_params = model_params or {}

    def to_dict(self) -> Dict[str, Any]:
        """
        Returns a dictionary representation of the request. Only includes
        the elements that are required by the OpenAI API. Model parameters
        are flattened into the top-level dictionary.
        """
        return {
            "model": self.model.name,
            "messages": self.chat.to_list(),
            "functions": self.functions,
            "function_call": self.function_call,
            **self.model_params,
        }

    def count_tokens(self) -> int:
        """
        Counts the number of tokens that will be used as input to the model.
        This includes the chat messages and the function call. Note that
        the output tokens are not counted and depend on the model's response.
        """
        chat_tokens = self.chat.count_tokens()
        function_tokens = num_tokens_from_text(json.dumps(self.function_call))

        return chat_tokens + function_tokens

    def estimate_input_cost_usd(self) -> float:
        """
        Estimates the cost of the request in USD.
        """
        num_input_tokens = self.count_tokens()

        input_cost_usd = num_input_tokens * self.model.input_token_price_per_1k / 1000

        return input_cost_usd


def build_binpacked_requests(
    model: Model,
    function: FunctionDef,
    system_message: str,
    texts: List[str],
    max_tokens_per_request: Optional[int] = None,
    max_texts_per_chat: Optional[int] = None,
    binpacking_function: Callable = binpack_texts_in_order,
    formatter_function: Callable = format_texts_as_json,
    encoding_name: str = "cl100k_base",
    long_text_handling: str = "error",
    model_params: Optional[Dict[str, Any]] = None,
) -> List[ChatCompletionRequest]:
    """
    Builds a list of ChatCompletionRequests from a list of texts.

    The list can then be passed to processor.process_api_requests().

    Args:
        model: The model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
        system_message: The message to include at the beginning of each chat.
        texts: A list of texts to binpack into chats.
        max_tokens_per_chat: The maximum number of tokens allowed per chat. Defaults
            to 90% of the model's context size.
        max_texts_per_chat: The maximum number of texts allowed per chat. Defaults
            to None, which means there is no limit.
        binpacking_function: The function to use for binpacking.
            Must take a list of texts and return a list of lists of texts.
            Defaults to binpack_texts_in_order().
        formatter_function: The function to use for formatting the texts.
            Must take a list of texts and return a single string.
            Defaults to format_texts_as_json().
        encoding_name: The name of the encoding to use for tokenization.
            Defaults to "cl100k_base".
        long_text_handling: Passed to the binpacking function. Defaults to
            "error", which means that an error will be raised if a text is too
            long to fit in a single chat.
        model_params: Additional keyword arguments to pass to the OpenAI API. See
            https://platform.openai.com/docs/api-reference/completions/create

    Returns:
        A list of ChatCompletionRequests.
    """
    if max_tokens_per_request is None:
        max_tokens_per_request = int(model.context_size * 0.9)

    # The system message counts towards the token limit
    static_tokens = num_tokens_from_text(system_message) + num_tokens_from_text(
        json.dumps(function)
    )

    max_tokens_per_chat = max_tokens_per_request - static_tokens

    # Binpack the texts into chats
    text_bins = binpacking_function(
        texts=texts,
        max_tokens=max_tokens_per_chat,
        max_texts=max_texts_per_chat,
        encoding_name=encoding_name,
        formatter_function=formatter_function,
        long_text_handling=long_text_handling,
    )

    requests = []

    for text_bin in text_bins:
        # Create a chat from the bin
        messages = [ChatMessage("system", system_message)]
        messages.append(ChatMessage("user", formatter_function(text_bin)))

        chat = Chat(messages)

        request = ChatCompletionRequest(
            chat=chat,
            model=model,
            function=function,
            model_params=model_params,
        )

        requests.append(request)

    return requests
