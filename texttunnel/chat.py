from typing import Dict, List, Optional
from dataclasses import dataclass
import tiktoken
import json


FunctionDef = Dict[str, str]


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a string.
    Args:
        text: The text to count tokens in.
        encoding_name: The name of the token encoding to use. Defaults to "cl100k_base".

    Returns:
        The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def binpack_texts_in_order(
    texts: List[str],
    max_tokens: int,
    max_texts: Optional[int] = None,
    encoding_name: str = "cl100k_base",
) -> List[List[str]]:
    """
    Binpacks a list of texts into a list of lists of texts, such that each list of texts
    has a total number of tokens less than or equal to max_tokens and each list of texts
    has a number of texts less than or equal to max_texts.

    The binpacking uses a naive greedy algorithm that maintains the order of the texts.

    Args:
        texts: The texts to binpack. Empty texts are accepted, counted as 0 tokens
            each and count against max_texts.
        max_tokens: The maximum number of tokens per list of texts.
        max_texts: The maximum number of texts per list of texts. Defaults to None, which
            means that there is no limit on the number of texts per list of texts.
        encoding_name: The name of the encoding to use. Defaults to "cl100k_base".

    Returns:
        A list of lists of texts. The order of the texts is preserved.
    """

    if not max_texts:
        max_texts = len(texts)

    # Count the number of tokens in each text
    # Don't use num_tokens_from_string() because we don't want to use get_encoding() for each text
    encoder = tiktoken.get_encoding(encoding_name)
    num_tokens_list = [len(encoder.encode(text)) for text in texts]

    # Binpack the texts
    # Initialize the first bin
    bins = []
    current_bin = []
    current_bin_tokens = 0
    current_bin_texts = 0

    for i, (text, num_tokens) in enumerate(zip(texts, num_tokens_list)):
        if num_tokens > max_tokens:
            raise ValueError(f"Text at index {i} has more than {max_tokens} tokens.")

        # Check if we need to start a new bin
        if (
            current_bin_tokens + num_tokens > max_tokens
            or current_bin_texts == max_texts
        ):
            # Start a new bin
            bins.append(current_bin)
            current_bin = []
            current_bin_tokens = 0
            current_bin_texts = 0

        # Add to the current bin
        current_bin.append(text)
        current_bin_tokens += num_tokens
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
        return {"role": self.role, "content": self.content}

    def num_tokens_from_string(self) -> int:
        return num_tokens_from_string(self.content)


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

    def to_list(self) -> List[Dict[str, str]]:
        return [message.to_dict() for message in self.messages]

    def num_tokens_from_string(self) -> int:
        return sum(message.num_tokens_from_string() for message in self.messages)


@dataclass
class Model:
    """
    Information about an OpenAI ChatCompletion model.
    Check prices here: https://openai.com/pricing

    Note that rate limits differ between OpenAI accounts.
    Check them here: https://platform.openai.com/account/rate-limits

    Args:
        - name: The name of the model, e.g. "gpt-3.5-turbo".
        - context_size: The maximum number of tokens that can be passed to the model.
        - input_token_price_per_1k: The price in USD per 1000 tokens for input.
        - output_token_price_per_1k: The price in USD per 1000 tokens for output.
        - tokens_per_minute: The maximum number of tokens that can be processed per minute.
        - requests_per_minute: The maximum number of requests that can be made per minute.
    """

    name: str
    context_size: int
    input_token_price_per_1k: float
    output_token_price_per_1k: float
    tokens_per_minute: int
    requests_per_minute: int


class ChatCompletionRequest:
    """
    Defines a request for a chat completion.

    Args:
        chat: The chat to which the assistant should respond with a function call.
        model: The name of the OpenAI ChatCompletion model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
    """

    def __init__(
        self,
        chat: Chat,
        model: Model,
        function: FunctionDef,
    ):
        self.chat = chat
        self.model = model

        self.function = function

        # Force the model to use a function call
        self.functions = [function]
        self.function_call = {"name": function["name"]}

    def to_dict(self) -> Dict[str, object]:
        return {
            "model": self.model.name,
            "messages": self.chat.to_list(),
            "functions": self.functions,
            "function_call": self.function_call,
        }

    def num_tokens_from_string(self) -> int:
        chat_tokens = self.chat.num_tokens_from_string()
        function_tokens = num_tokens_from_string(json.dumps(self.function_call))

        return chat_tokens + function_tokens

    def estimate_input_cost_usd(self) -> float:
        """
        Estimates the cost of the request in USD.
        """
        num_input_tokens = self.num_tokens_from_string()

        input_cost_usd = num_input_tokens * self.model.input_token_price_per_1k / 1000

        return input_cost_usd
