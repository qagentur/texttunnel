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
            Note that this may differ between OpenAI accounts. Override the default
            models' values with your own values.
        requests_per_minute: The maximum number of requests that can be made per minute.
            Note that this may differ between OpenAI accounts. Override the default
            models' values with your own values.
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


# Look up information on models, pricing and rate limits:
# https://platform.openai.com/docs/models/overview
# https://openai.com/pricing
# https://platform.openai.com/account/rate-limits

GPT_4 = Model(
    name="gpt-4",
    context_size=8192,
    input_token_price_per_1k=0.03,
    output_token_price_per_1k=0.06,
    tokens_per_minute=10000,
    requests_per_minute=500,
)

GPT_4_0613 = Model(
    name="gpt-4-0613",
    context_size=8192,
    input_token_price_per_1k=0.03,
    output_token_price_per_1k=0.06,
    tokens_per_minute=10000,
    requests_per_minute=500,
)

GPT_4_32K = Model(
    name="gpt-4-32k",
    context_size=32768,
    input_token_price_per_1k=0.06,
    output_token_price_per_1k=0.12,
    tokens_per_minute=20000,
    requests_per_minute=500,
)

GPT_4_32K_0613 = Model(
    name="gpt-4-32k-0613",
    context_size=32768,
    input_token_price_per_1k=0.06,
    output_token_price_per_1k=0.12,
    tokens_per_minute=20000,
    requests_per_minute=500,
)

# legacy
GPT_4_0314 = Model(
    name="gpt-4-0314",
    context_size=8192,
    input_token_price_per_1k=0.03,
    output_token_price_per_1k=0.06,
    tokens_per_minute=10000,
    requests_per_minute=500,
)

# legacy
GPT_4_32K_0314 = Model(
    name="gpt-4-32k-0314",
    context_size=32768,
    input_token_price_per_1k=0.06,
    output_token_price_per_1k=0.12,
    tokens_per_minute=10000,
    requests_per_minute=500,
)

GPT_3_5_TURBO = Model(
    name="gpt-3.5-turbo",
    context_size=4096,
    input_token_price_per_1k=0.0015,
    output_token_price_per_1k=0.002,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)

GPT_3_5_TURBO_16K = Model(
    name="gpt-3.5-turbo-16k",
    context_size=16384,
    input_token_price_per_1k=0.003,
    output_token_price_per_1k=0.004,
    tokens_per_minute=180000,
    requests_per_minute=3500,
)

GPT_3_5_TURBO_0613 = Model(
    name="gpt-3.5-turbo-0613",
    context_size=4096,
    input_token_price_per_1k=0.0015,
    output_token_price_per_1k=0.002,
    tokens_per_minute=90000,
    requests_per_minute=3500,
)

GPT_3_5_TURBO_16K_0613 = Model(
    name="gpt-3.5-turbo-16k-0613",
    context_size=16384,
    input_token_price_per_1k=0.003,
    output_token_price_per_1k=0.004,
    tokens_per_minute=180000,
    requests_per_minute=3500,
)

# legacy
GPT_3_5_TURBO_0301 = Model(
    name="gpt-3.5-turbo-0301",
    context_size=4096,
    input_token_price_per_1k=0.0015,
    output_token_price_per_1k=0.002,
    tokens_per_minute=9000,
    requests_per_minute=3500,
)


class Parameters:
    """
    Set of parameters that can be passed to an API request.

    The parameters are explained in the OpenAI API documentation:
    https://platform.openai.com/docs/api-reference/chat/create

    Args:
        max_tokens: The maximum number of tokens to generate. Note:
            This can't be greater than the model's context size and should be at least
            long enough to fit the whole expected JSON output. This parameter is used
            to estimate the cost of the request.
        temperature: What sampling temperature to use, between 0 and 2.
            Higher values like 0.8 will make the output more random, while
            lower values like 0.2 will make it more focused and deterministic.
            Defaults to 0.0 because this package is designed for deterministic
            JSON-schema compliant output.
        presence_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on whether they appear in the text so far,
            increasing the model's likelihood to talk about new topics. Defaults to 0.0.
        frequency_penalty: Number between -2.0 and 2.0. Positive values penalize
            new tokens based on their existing frequency in the text so far,
            decreasing the model's likelihood to repeat the same line verbatim.
            Defaults to 0.0.
        seed: Integer seed for random number generation. Defaults to 42.

    Parameters that are not listed here are not supported by this package. The
    reason is that they're not relevant for the use case of this package.
    """

    def __init__(
        self,
        max_tokens: int,
        temperature: float = 0.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        seed: int = 42,
    ):
        if max_tokens < 1:
            raise ValueError("max_tokens must be positive")

        if temperature < 0 or temperature > 1:
            raise ValueError("temperature must be between 0 and 1")

        if frequency_penalty < -2 or frequency_penalty > 2:
            raise ValueError("frequency_penalty must be between -2 and 2")

        if presence_penalty < -2 or presence_penalty > 2:
            raise ValueError("presence_penalty must be between -2 and 2")

        self.max_tokens = max_tokens
        self.temperature = temperature
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty
        self.seed = seed

    def to_dict(self):
        """
        Returns:
            A dictionary representation of the parameters.
        """

        return {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "seed": self.seed,
        }
