from texttunnel import chat

GPT_3_5_TURBO = chat.Model(
    name="gpt-3.5-turbo",
    context_size=4096,
    input_token_price_per_1k=0.002,
    output_token_price_per_1k=0.004,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)

GPT_4 = chat.Model(
    name="gpt-4",
    context_size=8192,
    input_token_price_per_1k=0.03,
    output_token_price_per_1k=0.06,
    tokens_per_minute=40000,
    requests_per_minute=200,
)
