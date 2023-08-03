from typing import List


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
