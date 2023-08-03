from typing import Dict, List
from dataclasses import dataclass


FunctionDef = Dict[str, str]


class ChatMessage:
    """
    A chat message.
    """

    def __init__(self, role: str, content: str):
        assert role in ["system", "user", "assistant"]

        self.role = role
        self.content = content

    def to_dict(self):
        return {"role": self.role, "content": self.content}


class Chat:
    """
    A chat. Used to prompt a model for a response.
    The first message must be from the system, and the last message must be from the user.
    """

    def __init__(self, messages: List):
        assert len(messages) >= 2
        assert messages[0].role == "system"
        assert messages[-1].role == "user"

        self.messages = messages

    def to_list(self):
        return [message.to_dict() for message in self.messages]


@dataclass
class Model:
    model: str
    context_size: int
    input_token_price_per_1k: float
    output_token_price_per_1k: float
    tokens_per_minute: int
    requests_per_minute: int


class ChatCompletionRequest:
    """
    Defines a request for a chat completion.

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

    def to_dict(self):
        return {
            "model": self.model,
            "messages": self.chat.to_list(),
            "functions": self.functions,
            "function_call": self.function_call,
        }
