from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from jsonschema import Draft7Validator, exceptions
import tiktoken
import json

from texttunnel.models import Model
from texttunnel import utils

FunctionDef = Dict[str, str]


def is_valid_function_def(function: FunctionDef) -> bool:
    """
    Checks if a function definition is valid for use in a ChatCompletionRequest.
    Note that the parameter properties are not validated to allow for custom properties.

    Args:
        function: The function definition to validate.
    """
    base_schema = {
        "name": {"type": "string"},
        "description": {"type": "string"},
        "parameters": {
            "type": "object",
            "properties": {"type": "object"},
        },
        "required": ["name", "parameters"],
    }

    try:
        Draft7Validator(base_schema).validate(function)
    except exceptions.ValidationError:
        print(f"Validation error: {exceptions.ValidationError}")
        return False

    return True


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
        """
        Returns a dict representation of the message.
        """
        return {"role": self.role, "content": self.content}

    def count_tokens(self) -> int:
        """
        Returns the number of tokens in the message.
        """
        return utils.num_tokens_from_text(self.content)


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


@dataclass
class Model:
    """
    Information about an OpenAI ChatCompletion model.
    Check prices here: https://openai.com/pricing

    Note that rate limits differ between OpenAI accounts.
    Check them here: https://platform.openai.com/account/rate-limits

    Args:
        name: The name of the model, e.g. "gpt-3.5-turbo".
        context_size: The maximum number of tokens that can be passed to the model.
        input_token_price_per_1k: The price in USD per 1000 tokens for input.
        output_token_price_per_1k: The price in USD per 1000 tokens for output.
        tokens_per_minute: The maximum number of tokens that can be processed per minute.
        requests_per_minute: The maximum number of requests that can be made per minute.
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

        if not is_valid_function_def(function):
            raise ValueError("Invalid function definition.")

        self.function = function

        # Force the model to use a function call
        self.functions = [function]
        self.function_call = {"name": function["name"]}

        self.model_params = model_params or {}

        # Check that the inputs fit into the context size
        num_input_tokens = self.count_tokens()
        if num_input_tokens > self.model.context_size:
            raise ValueError(
                f"Input tokens ({num_input_tokens}) exceed the context size ({self.model.context_size})."
            )

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
        function_tokens = utils.num_tokens_from_text(json.dumps(self.function_call))

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
    binpacking_function: Callable = utils.binpack_texts_in_order,
    formatter_function: Callable = utils.format_texts_as_json,
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
    static_tokens = utils.num_tokens_from_text(
        system_message
    ) + utils.num_tokens_from_text(json.dumps(function))

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
