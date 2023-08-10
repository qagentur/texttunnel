# %%
from texttunnel import chat, models, processor
from diskcache import Cache

# Create a cache to store the results of the requests
# When this script is run again, the results will be loaded from the cache
cache = Cache("mycache")

# Look up information on models, pricing and rate limits:
# https://platform.openai.com/docs/models/overview
# https://openai.com/pricing
# https://platform.openai.com/account/rate-limits
GPT_3_5_TURBO = models.Model(
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
            "answers": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "sentiment": {"type": "string"},
                    },
                    "required": ["id", "sentiment"],
                },
            },
        },
        "required": ["answers"],
    },
}

system_message = "You are a sentiment analysis expert. Analyze the following statements as positive or negative."

requests = chat.build_binpacked_requests(
    texts=input_texts,
    function=function,
    model=GPT_3_5_TURBO,
    system_message=system_message,
    model_params={
        "temperature": 0.0,
    },  # no randomness in the model's output
)

# %%
# Estimate the cost of the requests
cost_usd = sum([r.estimate_input_cost_usd() for r in requests])
print(f"Estimated cost of input tokens: ${cost_usd:.4f}")

# %%
# Requires that the OPENAI_API_KEY environment variable is set.
responses = processor.process_api_requests(
    requests=requests,
    logging_level=20,  # log INFO and above
    cache=cache,  # use diskcache to cache API responses
    api_key="abc",
)

cache.close()

# %%
results = processor.parse_responses(responses=responses)

for text, answer in zip(input_texts, results[0]["answers"]):
    print(f"{text}: {answer['sentiment']}")

# %%
