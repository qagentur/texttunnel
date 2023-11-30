import json
from typing import Any, Callable, Dict, List, Optional

import tiktoken
from jsonschema import Draft7Validator, exceptions

from texttunnel import utils
from texttunnel.models import Model, Parameters

FunctionDef = Dict[str, str]


def is_valid_function_def(function: FunctionDef) -> bool:
    """
    Checks if a function definition is valid for use in a ChatCompletionRequest.
    Note that the parameter properties are not validated to allow for custom properties.

    Check the OpenAI API documentation for more information:
    https://platform.openai.com/docs/guides/gpt/function-calling

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
        Return the number of tokens used.
        Note that this depends on the model used. Models that are not versioned
        with a date can change over time, causing an inaccurate token count
        by this function.

        Args:
            model: The name of the model to use. Defaults to "gpt-3.5-turbo-0613".

        Returns:
            The number of tokens used.
        """

        # See reference implementation:
        # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb

        # Note that the reference implementation uses varying numbers of tokens
        # for tokens_per_message depending on model. At the time of writing,
        # only gpt-3.5-turbo-0301 differs from the rest by one token per message.
        # To allow any OpenAI model to be used, we use 3 tokens per message.
        # This causes an underestimation of the token count when using gpt-3.5-turbo-0301.

        encoding = tiktoken.get_encoding("cl100k_base")
        tokens_per_message = 3
        tokens_per_name = 1
        num_tokens = 0

        for message in self.messages:
            num_tokens += tokens_per_message
            num_tokens += len(encoding.encode(message.content))
            if message.role == "name":
                num_tokens += tokens_per_name
        num_tokens += 3  # every reply is primed with <|start|>assistant<|message|>

        return num_tokens


class ChatCompletionRequest:
    """
    Defines a request for a chat completion.

    Args:
        chat: The chat to which the assistant should respond with a function call.
        model: The name of the OpenAI ChatCompletion model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
        params: Object of class Parameters. See models.Parameters for details.
    """

    def __init__(
        self,
        chat: Chat,
        model: Model,
        function: FunctionDef,
        params: Parameters,
    ):
        self.chat = chat
        self.model = model

        if not is_valid_function_def(function):
            raise ValueError("Invalid function definition.")

        self.function = function

        # Force the model to use a function call
        self.functions = [function]
        self.function_call = {"name": function["name"]}

        if params.max_tokens > self.model.context_size:
            raise ValueError(
                f"""
                max_tokens ({params.max_tokens}) exceeds the context
                size of the model ({self.model.context_size}).
                """
            )

        self.params = params

        # Check that the inputs fit into the context size and leaves
        # enough space for the output
        num_prompt_tokens = self.count_prompt_tokens()
        num_completion_tokens = self.count_completion_tokens()
        num_total_tokens = num_prompt_tokens + num_completion_tokens

        if num_total_tokens > self.model.context_size:
            raise ValueError(
                f"""
                Total number of tokens ({num_total_tokens}) exceeds the context
                size of the model ({self.model.context_size}). Input tokens:
                {num_prompt_tokens}. Output tokens: {num_completion_tokens}.
                Context size: {self.model.context_size}.
                """
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
            **self.params.to_dict(),
        }

    def get_hash(self) -> str:
        """
        Returns the hash of the request. Can be used as a cache key.
        """
        return utils.hash_dict(self.to_dict())

    def count_prompt_tokens(self) -> int:
        """
        Counts the number of tokens that will be used as input to the model.
        This includes the chat messages and the function call.
        """
        chat_tokens = self.chat.count_tokens()
        function_tokens = utils.num_tokens_from_text(json.dumps(self.functions[0]))

        return chat_tokens + function_tokens

    def count_completion_tokens(self) -> int:
        """
        Counts the number of tokens that will be used as output of the model.
        Assumes that the model will return the maximum number of tokens allowed
        by the max_tokens parameter.
        """

        return self.params.max_tokens

    def count_total_tokens(self) -> int:
        """
        Counts the total number of tokens that will be used as input and output
        of the model. Assumes that the model will return the maximum number of
        tokens allowed by the max_tokens parameter.
        """
        return self.count_prompt_tokens() + self.count_completion_tokens()

    def estimate_cost_usd(self) -> float:
        """
        Estimates the cost of the request in USD. Assumes that the model will
        return the maximum number of tokens allowed by the max_tokens parameter.
        The estimate is the upper bound on the cost, since the model may return
        fewer tokens than the maximum allowed.
        """

        input_cost_usd = (
            self.count_prompt_tokens() * self.model.input_token_price_per_1k / 1000
        )
        output_cost_usd = (
            self.count_completion_tokens() * self.model.output_token_price_per_1k / 1000
        )

        return input_cost_usd + output_cost_usd


def build_binpacked_requests(
    model: Model,
    function: FunctionDef,
    system_message: str,
    texts: List[str],
    params: Parameters,
    max_tokens_per_request: Optional[int] = None,
    max_texts_per_request: Optional[int] = None,
    binpacking_function: Callable = utils.binpack_texts_in_order,
    formatter_function: Callable = utils.format_texts_as_json,
    encoding_name: str = "cl100k_base",
    long_text_handling: str = "error",
) -> List[ChatCompletionRequest]:
    """
    Builds a list of ChatCompletionRequests from a list of texts.
    If possible, multiple texts will be combined into a single ChatCompletionRequest.
    This can reduce the number of tokens spent on overheads like the system message
    and function definition.

    The requests can then be passed to processor.process_api_requests().

    Args:
        model: The model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
        system_message: The message to include at the beginning of each chat.
        texts: A list of texts to binpack into chats. Duplicates are not allowed.
        params: Object of class Parameters. See models.Parameters for details.
        max_tokens_per_request: The maximum number of tokens allowed in one request.
            Defaults to 90% of the model's context size. The 10% buffer makes
            sure that mistakes in token counting don't cause the request to fail.
        max_texts_per_request: The maximum number of texts allowed in one request.
            Defaults to None, which means there is no limit.
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

    Returns:
        A list of ChatCompletionRequests.
    """
    if len(set(texts)) != len(texts):
        # Downstream code assumes that each request has a unique hash
        # Duplicate texts would cause the requests to have the same hash
        # Plus it's probably a mistake and would waste money
        raise ValueError("Duplicate texts found. Please remove duplicates.")

    if max_tokens_per_request is None:
        max_tokens_per_request = int(model.context_size * 0.9)

    # System message and function definition count towards the token limit
    overheads = [system_message, json.dumps(function)]
    static_tokens = sum([utils.num_tokens_from_text(text) for text in overheads])

    # Calculate the maximum number of tokens left for the chat,
    # after accounting for the overheads and the output tokens
    max_tokens_per_chat = max_tokens_per_request - static_tokens - params.max_tokens

    # Binpack the texts into chats
    text_bins = binpacking_function(
        texts=texts,
        max_tokens_per_bin=max_tokens_per_chat,
        max_texts_per_bin=max_texts_per_request,
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
            params=params,
        )

        requests.append(request)

    return requests


def build_requests(
    model: Model,
    function: FunctionDef,
    system_message: str,
    texts: List[str],
    params: Parameters,
    encoding_name: str = "cl100k_base",
    long_text_handling: str = "error",
) -> List[ChatCompletionRequest]:
    """
    Builds a list of ChatCompletionRequests from a list of texts.
    The requests can then be passed to processor.process_api_requests().

    Args:
        model: The model to use for completion.
        function: The function definition to use for the assistant's response.
            Must be a dictionary that describes a valid JSON schema.
            See https://platform.openai.com/docs/guides/gpt/function-calling
        system_message: The message to include at the beginning of each chat.
        params: Object of class Parameters. See models.Parameters for details.
        texts: A list of texts to binpack into chats. Duplicates are not allowed.
        encoding_name: The name of the encoding to use for tokenization.
            Defaults to "cl100k_base".
        long_text_handling: Passed to the binpacking function. Defaults to
            "error", which means that an error will be raised if a text is too
            long to fit in a single chat.

    Returns:
        A list of ChatCompletionRequests.
    """

    return build_binpacked_requests(
        model=model,
        function=function,
        system_message=system_message,
        texts=texts,
        params=params,
        max_tokens_per_request=None,
        max_texts_per_request=1,
        binpacking_function=utils.binpack_texts_in_order,
        formatter_function=utils.format_texts_with_spaces,
        encoding_name=encoding_name,
        long_text_handling=long_text_handling,
    )
