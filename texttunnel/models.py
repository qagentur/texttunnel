from dataclasses import dataclass


@dataclass
class Model:
    """
    Information about an OpenAI ChatCompletion model.
    Check prices here: https://openai.com/pricing

    Note that rate limits differ between OpenAI accounts.
    Check them here: https://platform.openai.com/account/rate-limits

    Args:
        name: The name of the model, e.g. "gpt-3.5-turbo".
        context_size: The maximum number of tokens that can be passed to the model.
        input_token_price_per_1k: The price in USD per 1000 tokens for input.
        output_token_price_per_1k: The price in USD per 1000 tokens for output.
        tokens_per_minute: The maximum number of tokens that can be processed per minute.
        requests_per_minute: The maximum number of requests that can be made per minute.
    """

    name: str
    context_size: int
    input_token_price_per_1k: float
    output_token_price_per_1k: float
    tokens_per_minute: int
    requests_per_minute: int

    def __post_init__(self):
        # Check that inputs are positive

        for arg in [
            "context_size",
            "input_token_price_per_1k",
            "output_token_price_per_1k",
            "tokens_per_minute",
            "requests_per_minute",
        ]:
            if getattr(self, arg) < 0:
                raise ValueError(f"{arg} must be positive")


GPT_3_5_TURBO = Model(
    name="gpt-3.5-turbo",
    context_size=4096,
    input_token_price_per_1k=0.002,
    output_token_price_per_1k=0.004,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)

GPT_4 = Model(
    name="gpt-4",
    context_size=8192,
    input_token_price_per_1k=0.03,
    output_token_price_per_1k=0.06,
    tokens_per_minute=40000,
    requests_per_minute=200,
)
