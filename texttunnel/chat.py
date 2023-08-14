# --------------------------------------------------------------------------------
# This file includes a function adapted from: openai-cookbook
# Original source code: https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
# Copyright (c) 2023 OpenAI

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
from typing import Any, Callable, Dict, List, Optional

from jsonschema import Draft7Validator, exceptions
import tiktoken

from texttunnel import utils
from texttunnel.models import Model

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

    def count_tokens(
        self,
        model: str = "gpt-3.5-turbo-0613",
        show_changing_model_warning: bool = False,
    ) -> int:
        """
        Return the number of tokens used.
        Note that this depends on the model used. Models that are not versioned
        with a date can change over time, causing an inaccurate token count
        by this function.

        Args:
            model: The name of the model to use. Defaults to "gpt-3.5-turbo-0613".
            show_changing_model_warning: Whether to print a warning if a model
                is used that may change over time. Defaults to False.

        Returns:
            The number of tokens used.
        """
        # This function was adapted from openai-cookbook

        try:
            encoding = tiktoken.encoding_for_model(model)
        except KeyError:
            if show_changing_model_warning:
                print("Warning: model not found. Using cl100k_base encoding.")
            encoding = tiktoken.get_encoding("cl100k_base")

        if model == "gpt-3.5-turbo":
            if show_changing_model_warning:
                print(
                    "Warning: gpt-3.5-turbo may update over time. Returning num tokens assuming gpt-3.5-turbo-0613."
                )
            model = "gpt-3.5-turbo-0613"

        if model == "gpt-4":
            if show_changing_model_warning:
                print(
                    "Warning: gpt-4 may update over time. Returning num tokens assuming gpt-4-0613."
                )
            model = "gpt-4-0613"

        if model in {
            "gpt-3.5-turbo-0613",
            "gpt-3.5-turbo-16k-0613",
            "gpt-4-0314",
            "gpt-4-32k-0314",
            "gpt-4-0613",
            "gpt-4-32k-0613",
        }:
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-3.5-turbo-0301":
            tokens_per_message = (
                4  # every message follows <|start|>{role/name}\n{content}<|end|>\n
            )
            tokens_per_name = -1  # if there's a name, the role is omitted
        else:
            raise NotImplementedError(
                f"""num_tokens_from_messages() is not implemented for model {model}. See https://github.com/openai/openai-python/blob/main/chatml.md for information on how messages are converted to tokens."""
            )
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
        max_tokens: The maximum number of tokens allowed in the completion.
            Defaults to 128.
        model_params: Additional keyword arguments to pass to the OpenAI API. See
            https://platform.openai.com/docs/api-reference/completions/create
    """

    def __init__(
        self,
        chat: Chat,
        model: Model,
        function: FunctionDef,
        max_tokens: int = 128,
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

        if model_params is None:
            model_params = {"max_tokens": max_tokens}
        else:
            model_params["max_tokens"] = max_tokens

        self.model_params = model_params

        # Check that the inputs fit into the context size
        num_input_tokens = self.count_input_tokens()
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

    def get_hash(self) -> str:
        """
        Returns the hash of the request. Can be used as a cache key.
        """
        return utils.hash_dict(self.to_dict())

    def count_input_tokens(self) -> int:
        """
        Counts the number of tokens that will be used as input to the model.
        This includes the chat messages and the function call.
        """
        chat_tokens = self.chat.count_tokens(model=self.model.name)
        function_tokens = utils.num_tokens_from_text(json.dumps(self.function_call))

        return chat_tokens + function_tokens

    def count_output_tokens(self) -> int:
        """
        Counts the number of tokens that will be used as output of the model.
        Assumes that the model will return the maximum number of tokens allowed
        by the max_tokens parameter.
        """

        n = self.model_params.get("n", 1)  # number of completions
        return self.model_params["max_tokens"] * n

    def count_total_tokens(self) -> int:
        """
        Counts the total number of tokens that will be used as input and output
        of the model. Assumes that the model will return the maximum number of
        tokens allowed by the max_tokens parameter.
        """
        return self.count_input_tokens() + self.count_output_tokens()

    def estimate_cost_usd(self) -> float:
        """
        Estimates the cost of the request in USD. Assumes that the model will
        return the maximum number of tokens allowed by the max_tokens parameter.
        """

        input_cost_usd = (
            self.count_input_tokens() * self.model.input_token_price_per_1k / 1000
        )
        output_cost_usd = (
            self.count_output_tokens() * self.model.output_token_price_per_1k / 1000
        )

        return input_cost_usd + output_cost_usd


def build_binpacked_requests(
    model: Model,
    function: FunctionDef,
    system_message: str,
    texts: List[str],
    max_tokens_per_request: Optional[int] = None,
    max_texts_per_request: Optional[int] = None,
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
        max_texts_per_request: The maximum number of texts allowed per chat. Defaults
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
            model_params=model_params,
        )

        requests.append(request)

    return requests
