# texttunnel: Efficient text processing with GPT-3.5 and GPT-4

This package provides a simple interface to the GPT-3.5 and GPT-4 models for natural language processing pipelines. It places an emphasis on efficiency and tidyness for a specific use case:

You have a corpus of text data that you want to process with GPT-3.5 or GPT-4 and you want the model to conduct extractive NLP such as classification, named entity, recognition, translation, summarization, question answering or sentiment analysis.

🚧 Status: This package is in early development and is not yet ready for use. 🚧

Features:

- 📄 Defining the output schema using [JSON schema](https://json-schema.org) and OpenAI's function calling schema.
- ✔️ Input validation.
- ✅ Output validation to confirm that the output from OpenAI's API matches the expected schema.
- 🚀 Efficient batching of text data for inference, packing multiple input texts into a request to OpenAI's API.
- 🚦 Asynchronous requests to OpenAI's API while observing the rate limits of the API.
- 💰 Cost estimation before sending requests to OpenAI's API.
- 💾 Caching of requests via [diskcache](https://github.com/grantjenks/python-diskcache). Never pay for the same request twice.
- 📝 Logging of requests via [logging](https://docs.python.org/3/library/logging.html).

## Installation

## Usage

Create an account on [OpenAI](https://openai.com) and get an API key. Set it as an environment variable called `OPENAI_API_KEY`.

Check the [examples](examples) directory for examples of how to use this package.

## Development

Run tests with `poetry run pytest -v`

To get started with development, follow these steps:

- clone the repository
- install [poetry](https://python-poetry.org/docs/) if you don't have it yet
- navigate to the project folder
- run `poetry install` to install the dependencies
- run the tests with `poetry run pytest -v`

This project uses [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings.
