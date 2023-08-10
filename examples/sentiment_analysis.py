# %%
import texttunnel.chat
import texttunnel.models
import texttunnel.processor

# Look up information on models, pricing and rate limits:
# https://platform.openai.com/docs/models/overview
# https://openai.com/pricing
# https://platform.openai.com/account/rate-limits
GPT_3_5_TURBO = texttunnel.chat.Model(
    name="gpt-3.5-turbo",
    context_size=4096,
    input_token_price_per_1k=0.002,
    output_token_price_per_1k=0.004,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)

# Texts that we'd like to know the sentiment of
input_texts = [
    "I love sunshine",
    "I don't like rain",
]

# Describe the output format that we'd like to receive,
# using JSON Schema
function = {
    "name": "sentiment_analysis",
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
        model=GPT_3_5_TURBO,
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
