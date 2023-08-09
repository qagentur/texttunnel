import pytest

from texttunnel import models

def test_model_errors_on_negative():
    with pytest.raises(ValueError):
        models.Model(
            name="gpt-3.5-turbo",
            context_size=0,
            input_token_price_per_1k=-1,
            output_token_price_per_1k=0.004,
            tokens_per_minute=90000,
            requests_per_minute=3500,
        )
