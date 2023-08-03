from texttunnel.chat import ChatMessage, Chat

def test_chat():
    chat = Chat(
        messages=[
            ChatMessage(
                role="system",
                content="You are a helpful assistant.",
            ),
            ChatMessage(
                role="user",
                content="Hello, world!",
            ),
        ]
    )

    assert chat.messages[0].role == "system"
