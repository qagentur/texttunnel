# %%
import texttunnel.chat
import texttunnel.models
import texttunnel.processor

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

requests = texttunnel.chat.build_binpacked_requests(
    texts=input_texts,
    function=function,
    model=texttunnel.models.GPT_3_5_TURBO,
    system_message=system_message,
    kwargs={
        "temperature": 0.0,
    },  # no randomness in the model's output
)

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

for text, answer in zip(input_texts, results[0]["answers"]):
    print(f"{text}: {answer['sentiment']}")

# %%
