# texttunnel: Efficient text processing with GPT-3.5 and GPT-4

<div align="center">
  <img src="https://github.com/qagentur/texttunnel/assets/25177095/411ad918-d054-4d19-aef5-1dba9136db33" width="65%" />
</div>

This package offers a straightforward interface for integrating the GPT-3.5 and GPT-4 models into your natural language processing pipelines. It is optimally designed for the following scenario:

Suppose you possess a corpus of text data that you want to analyze using the GPT-3.5 or GPT-4 models. The goal is to perform extractive NLP tasks such as classification, named entity recognition, translation, summarization, question answering, or sentiment analysis. In this context, the package prioritizes efficiency and tidiness to provide you streamlined results.

Features:

- 📄 Output Schema: Utilizes [JSON Schema](https://json-schema.org) alongside OpenAI's function calling schema to define the output data structure.
- ✔️ Input Validation: Ensures well-structured and error-free API requests by validating input data.
- ✅ Output Validation: Checks the response data from OpenAI's API against the expected schema to maintain data integrity.
- 🚦 Asynchronous Requests: Facilitates speedy data processing by sending simultaneous requests to OpenAI's API, while staying within API rate limits.
- 🚀 Efficient Batching: Supports bulk processing by packing multiple input texts into a single request for the OpenAI's API.
- 💰 Cost Estimation: Aims for transparency in API utilization cost by providing cost estimates before sending API requests.
- 💾 Caching: Uses [aiohttp_client_cache](https://github.com/requests-cache/aiohttp-client-cache) to avoid redundant requests and reduce cost by caching previous requests. Supports SQLite, MongoDB, DynamoDB and Redis cache backends.
- 📝 Request Logging: Implements Python's native [logging](https://docs.python.org/3/library/logging.html) framework for tracking and logging all API requests.

Note that this package only works with [function calling](https://platform.openai.com/docs/guides/function-calling) and only with the OpenAI API. If you're looking for a more flexible solution, consider [instructor](https://github.com/jxnl/instructor) and [litellm](https://github.com/BerriAI/litellm). You might also consider using the [OpenAI Batch API](https://platform.openai.com/docs/api-reference/batch) as it offers savings compared to synchronous API calls.

⚠️ **Maintenance mode**: At this time no new features or enhancements are being developed. Only critical bugfixes will be made.

## Installation

The package is available on [PyPI](https://pypi.org/project/texttunnel/). To install it, run:
  
```bash
pip install texttunnel
```

or via poetry:

```bash
poetry add texttunnel
```

**Note**: If you want to use caching, you need to install the aiohttp_client_cache extras. Please refer to the [aiohttp_client_cache](https://github.com/requests-cache/aiohttp-client-cache#quickstart) documentation for more information.

## Usage

Check the docs: [https://qagentur.github.io/texttunnel/](https://qagentur.github.io/texttunnel/)

Create an account on [OpenAI](https://openai.com) and get an API key. Set it as an environment variable called `OPENAI_API_KEY`.

Check the [examples](examples) directory for examples of how to use this package.

If your account has been granted higher rate limits than the ones configured in the models module, you can override the default attributes of the Model class instances. See documentation of the models package module.

## Development

To get started with development, follow these steps:

- clone the repository
- install [poetry](https://python-poetry.org/docs/) if you don't have it yet
- navigate to the project folder
- run `poetry install` to install the dependencies
- run the tests with `poetry run pytest -v`

This project uses [Google-style](https://google.github.io/styleguide/pyguide.html#381-docstrings) docstrings and [black](https://github.com/psf/black) formatting. The docs are automatically built based on the docstrings.
