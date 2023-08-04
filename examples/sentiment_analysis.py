# %%
import texttunnel.chat
import texttunnel.models
import texttunnel.processor

# Texts that we'd like to know the sentiment of
input_texts = [
    "I love sunshine",
    "I don't like rain",
]

# Sentiment analysis function
function = {
    "name": "function_name",
    "parameters": {
        "type": "object",
        "properties": {
            "sentiment": {"type": "string", "enum": ["positive", "negative"]}
        },
    },
}

requests = []

for text in input_texts:
    messages = texttunnel.chat.Chat(
        [
            texttunnel.chat.ChatMessage(
                role="system",
                content="You are a sentiment analysis expert.",
            ),
            texttunnel.chat.ChatMessage(
                role="user",
                content=f"Classify the following statement as positive or negative: {text}",
            ),
        ]
    )

    request = texttunnel.chat.ChatCompletionRequest(
        model=texttunnel.models.GPT_3_5_TURBO,
        chat=messages,
        function=function,
    )

    requests.append(request)

# %%
# Estimate the cost of the requests
cost_usd = sum([r.estimate_input_cost_usd() for r in requests])
print(f"Estimated cost of input tokens: ${cost_usd:.4f}")

# %%
# Requires that the OPENAI_API_KEY environment variable is set.
responses = texttunnel.processor.process_api_requests(
    requests=requests,
    save_filepath="output.jsonl",
    keep_file=False,
    logging_level=50,  # only log errors
)

# %%
results = texttunnel.processor.parse_responses(responses=responses)

for text, sentiment in zip(input_texts, results):
    print(f"{text}: {sentiment}")

# %%
