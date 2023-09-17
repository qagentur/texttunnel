# Example of using the texttunnel package to perform text classification
# Uses the aprocess_api_call for more control over the event loop

# %%
import logging
from aiohttp_client_cache import SQLiteBackend
import asyncio
import nest_asyncio

from texttunnel import chat, models, processor

nest_asyncio.apply()  # to allow for asyncio.run() within Jupyter

# %%
# Create a SQLite cache to store the results of the requests
# When this script is run again, the results will be loaded from the cache
# Requires the additional package aiosqlite (pip install aiosqlite)
cache = SQLiteBackend(cache_name="openai_cache.sqlite", allowed_methods=["POST"])

logging.basicConfig(level=logging.WARN)
logging.getLogger("texttunnel").setLevel(logging.INFO)

# Texts that we'd like to know the sentiment of
input_texts = [
    "The 60% layout is great for travel, but I wish it had arrow keys",
    "The laser doesn't work on my glass desk. I'm returning it.",
    "I love the feel of the keys, but the RGB lighting is too bright.",
    "The scroll wheel is too sensitive. I keep scrolling past what I want.",
]

# Describe the output format that we'd like to receive,
# using JSON Schema
function = {
    "name": "text_classification",
    "parameters": {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": ["keyboard", "mouse"],
            },
        },
        "required": ["answers"],
    },
}

system_message = "Classify reviews by product category."

model = models.GPT_3_5_TURBO

requests = chat.build_requests(
    texts=input_texts,
    function=function,
    model=model,
    system_message=system_message,
    params=models.Parameters(max_tokens=50),
)

# %%
# Create an event loop and run the requests
# Alternatively, use processor.process_api_requests() and let it handle the event loop
loop = asyncio.get_event_loop()
responses = loop.run_until_complete(
    processor.aprocess_api_requests(requests, cache=cache)
)

# %%
# Display the results
results = [
    processor.parse_arguments(response=response)["category"] for response in responses
]

for text, result in zip(input_texts, results):
    print(f"{text}: {result}")

# %%
