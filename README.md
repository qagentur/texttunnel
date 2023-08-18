# texttunnel: Efficient text processing with GPT-3.5 and GPT-4

This package offers a straightforward interface for integrating the GPT-3.5 and GPT-4 models into your natural language processing pipelines. It is optimally designed for the following scenario:

Suppose you possess a corpus of text data that you want to analyze using the GPT-3.5 or GPT-4 models. The goal is to perform extractive NLP tasks such as classification, named entity recognition, translation, summarization, question answering, or sentiment analysis. In this context, the package prioritizes efficiency and tidiness to provide you streamlined results.

🚧 Status: This package is in beta. Don't use it in production. 🚧

Features:

- 📄 Output Schema: Utilizes [JSON Schema](https://json-schema.org) alongside OpenAI's function calling schema to define the output data structure.
- ✔️ Input Validation: Ensures well-structured and error-free API requests by validating input data.
- ✅ Output Validation: Checks the response data from OpenAI's API against the expected schema to maintain data integrity.
- 🚀 Efficient Batching: Supports bulk processing by packing multiple input texts into a single request for the OpenAI's API.
- 🚦 Asynchronous Requests: Facilitates speedy data processing by sending simultaneous requests to OpenAI's API, while maintaining API rate limits.
- 💰 Cost Estimation: Aims for transparency in API utilization cost by providing cost estimates before sending API requests.
- 💾 Disk Caching: Uses [diskcache](https://github.com/grantjenks/python-diskcache) to avoid redundant requests and reduce cost by caching previous requests.
- 📝 Request Logging: Implements Python's native [logging](https://docs.python.org/3/library/logging.html) framework for tracking and logging all API requests.

## Installation

The package is not yet available on PyPI. To install it, clone the repository and run `poetry install` or install it directly from GitHub using:

```bash
pip install git+https://github.com/qagentur/texttunnel.git
```

or adding this line to your pyproject.toml, if you're using poetry:

```toml
[tool.poetry.dependencies]
texttunnel = { git = "https://github.com/qagentur/texttunnel.git", branch = "main" }
```

## Usage

Check the docs: [https://qagentur.github.io/texttunnel/](https://qagentur.github.io/texttunnel/)

Create an account on [OpenAI](https://openai.com) and get an API key. Set it as an environment variable called `OPENAI_API_KEY`.

Check the [examples](examples) directory for examples of how to use this package.

If your account has been granted higher rate limits than the ones configured in the models module, you can override the default attributes of the Model class instances.

## Development

To get started with development, follow these steps:

- clone the repository
- install [poetry](https://python-poetry.org/docs/) if you don't have it yet
- navigate to the project folder
- run `poetry install` to install the dependencies
- run the tests with `poetry run pytest -v`

This project uses [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings and [black](https://github.com/psf/black) formatting. The docs are automatically built based on the docstrings.
