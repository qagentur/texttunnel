from texttunnel import chat

GPT_3_5_TURBO = chat.Model(
    name="gpt-3.5-turbo",
    context_size=4000,
    input_token_price_per_1k=0.002,
    output_token_price_per_1k=0.004,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)
