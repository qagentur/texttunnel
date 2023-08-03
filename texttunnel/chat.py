from typing import Dict, List
from dataclasses import dataclass
import tiktoken
import json


FunctionDef = Dict[str, str]


def count_tokens(text: str, encoding: str = "cl100k_base") -> int:
    """
    Returns the number of tokens in a string.

    Args:
        text: The text to count tokens in.
        encoding: The name of the encoding to use. Defaults to "cl100k_base".
    """

    encoder = tiktoken.get_encoding(encoding)
    num_tokens = len(encoder.encode(text))
    return num_tokens


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

    def count_tokens(self) -> int:
        return count_tokens(self.content)


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

    def count_tokens(self) -> int:
        return sum(message.count_tokens() for message in self.messages)


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

    def count_tokens(self) -> int:
        chat_tokens = self.chat.count_tokens()
        function_tokens = count_tokens(json.dumps(self.function_call))

        return chat_tokens + function_tokens
