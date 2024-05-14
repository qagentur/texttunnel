# Example of using the texttunnel package to perform named entity recognition
# Script requires that the OPENAI_API_KEY environment variable is set.

from texttunnel import chat, models, processor

# Texts that we'd like to extract entities from
input_texts = [
    "BioNTech SE is set to acquire InstaDeep, \
a Tunis-born and U.K.-based artificial intelligence \
(AI) startup, for up to £562 million",
    "The U.S. Food and Drug Administration (FDA) \
has approve Pfizer-BioNTech's COVID-19 vaccine for emergency use",
    "BioNTech founders, Dr. Ugur Sahin and Dr. Ozlem Tureci, \
receive prestigious award for their vaccine research",
]


# Describe the output format that we'd like to receive,
# using JSON Schema. We specify that we want to extract
# persons, organizations, and locations from the text.
function = {
    "name": "ner",
    "parameters": {
        "type": "object",
        "properties": {
            "persons": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "organizations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
            "locations": {
                "type": "array",
                "items": {
                    "type": "string",
                },
            },
        },
        "required": ["persons", "organizations", "locations"],
    },
}

system_message = "Extract named entities from a text."

model = models.GPT_4o

requests = chat.build_requests(
    texts=input_texts,
    function=function,
    model=model,
    system_message=system_message,
    params=models.Parameters(max_tokens=256),
)

# Estimate the cost of the requests
estimated_cost_usd = sum([r.estimate_cost_usd() for r in requests])
print(f"Estimated cost: ${estimated_cost_usd:.4f}")


responses = processor.process_api_requests(requests=requests)


results = [processor.parse_arguments(response=response) for response in responses]
print(results)

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
