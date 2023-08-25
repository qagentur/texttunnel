# %%
import logging

from diskcache import Cache

from texttunnel import chat, models, processor

# Create a cache to store the results of the requests
# When this script is run again, the results will be loaded from the cache
cache = Cache("mycache")

logging.basicConfig(level=logging.INFO)

# Look up information on models, pricing and rate limits:
# https://platform.openai.com/docs/models/overview
# https://openai.com/pricing
# https://platform.openai.com/account/rate-limits

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

model = models.GPT_3_5_TURBO

requests = chat.build_binpacked_requests(
    texts=input_texts,
    function=function,
    model=model,
    system_message=system_message,
    params=models.Parameters(max_tokens=50),
)

# %%
# Estimate the cost of the requests
estimated_cost_usd = sum([r.estimate_cost_usd() for r in requests])
print(f"Estimated cost: ${estimated_cost_usd:.4f}")

# %%
# Requires that the OPENAI_API_KEY environment variable is set.
responses = processor.process_api_requests(
    requests=requests,
    cache=cache,  # use diskcache to cache API responses
)

cache.close()

# %%
results = [processor.parse_arguments(response=response) for response in responses]

for text, answer in zip(input_texts, results[0]["answers"]):
    print(f"{text}: {answer['sentiment']}")

# %%
actual_cost_usd = sum(
    [
        processor.usage_to_cost(
            usage=processor.parse_token_usage(response=response),
            model=model,
        )
        for response in responses
    ]
)

print(f"Actual cost: ${actual_cost_usd:.4f}")

# %%
