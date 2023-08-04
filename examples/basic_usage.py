# %%
import texttunnel.chat
import texttunnel.models
import texttunnel.processor

messages = texttunnel.chat.Chat(
    [
        texttunnel.chat.ChatMessage(
            role="system",
            content="You are a sentiment analysis expert.",
        ),
        texttunnel.chat.ChatMessage(
            role="user",
            content="Classify the following statement as positive or negative: I love sunshine.",
        ),
    ]
)

function = {
    "name": "function_name",
    "parameters": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative"]}
        },
    },
}

request = texttunnel.chat.ChatCompletionRequest(
    model=texttunnel.models.GPT_3_5_TURBO,
    chat=messages,
    function=function,
)

requests = [request, request, request]

# Requires that the OPENAI_API_KEY environment variable is set.
texttunnel.processor.process_api_requests(
    requests=requests,
    save_filepath="output.jsonl",
)
