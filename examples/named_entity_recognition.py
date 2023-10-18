# Example of using the texttunnel package to perform named entity recognition
# Features the instructor package for easier definition of the function calling
# JSON schema. It also has binpacking to reduce the number of API calls

# %%
from enum import Enum
import logging
from typing import List

from aiohttp_client_cache import SQLiteBackend
from instructor import OpenAISchema

from pydantic import Field, BaseModel

from texttunnel import chat, models, processor

# Create a SQLite cache to store the results of the requests
# When this script is run again, the results will be loaded from the cache
# Requires the additional package aiosqlite (pip install aiosqlite)
cache = SQLiteBackend(cache_name="openai_cache.sqlite", allowed_methods=["POST"])

logging.basicConfig(level=logging.WARN)
logging.getLogger("texttunnel").setLevel(logging.INFO)

# Texts that we'd like to know the sentiment of
input_texts = [
    "Apple is looking at buying U.K. startup for $1 billion",
    "The European Union is very happy about the new trade deal with the United Kingdom",
    "Sam Altman pays $10 parking ticket",
]

# Describe the output format that we'd like to receive
# using a Pydantic model via the instructor package

class EntityLabel(Enum):
    ORG = "ORG"
    PERSON = "PERSON"
    MONEY = "MONEY"
    GPE = "GPE"

class Entity(BaseModel):
    text: str = Field(description="The entity extracted from the text")
    label: EntityLabel = Field(description="The type of entity extracted from the text")

class NamedEntities(OpenAISchema):
    entities: List[Entity] = Field(description="The entities extracted from the text")

print(NamedEntities.openai_schema)

system_message = "You are an NER model extracting ORG, PERSON, GPE and MONEY entities."

model = models.GPT_3_5_TURBO

requests = chat.build_requests(
    texts=input_texts,
    function=NamedEntities.openai_schema,
    model=model,
    system_message=system_message,
    params=models.Parameters(max_tokens=200),
)

# %%
# Estimate the cost of the requests
estimated_cost_usd = sum([r.estimate_cost_usd() for r in requests])
print(f"Estimated cost: ${estimated_cost_usd:.4f}")

# %%
# Requires that the OPENAI_API_KEY environment variable is set.
responses = processor.process_api_requests(
    requests=requests,
    cache=cache,
)

# %%
results = [processor.parse_arguments(response=response) for response in responses]

for text, entity_list in zip(input_texts, results[0]["entities"]):
    print(f"{text}: {entity_list}")

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
