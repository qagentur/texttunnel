.. texttunnel documentation master file, created by
   sphinx-quickstart on Fri Aug 18 14:26:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

texttunnel: Efficient text processing with GPT-3.5 and GPT-4
============================================================

This package offers a straightforward interface for integrating the GPT-3.5 and GPT-4 models into your natural language processing pipelines. It is optimally designed for the following scenario:

Suppose you possess a corpus of text data that you want to analyze using the GPT-3.5 or GPT-4 models. The goal is to perform extractive NLP tasks such as classification, named entity recognition, translation, summarization, question answering, or sentiment analysis. In this context, the package prioritizes efficiency and tidiness to provide you streamlined results.

Features:

- 📄 Output Schema: Utilizes JSON Schema alongside OpenAI's function calling schema to define the output data structure.
- ✔️ Input Validation: Ensures well-structured and error-free API requests by validating input data.
- ✅ Output Validation: Checks the response data from OpenAI's API against the expected schema to maintain data integrity.
- 🚀 Efficient Batching: Supports bulk processing by packing multiple input texts into a single request for the OpenAI's API.
- 🚦 Asynchronous Requests: Facilitates speedy data processing by sending simultaneous requests to OpenAI's API, while maintaining API rate limits.
- 💰 Cost Estimation: Aims for transparency in API utilization cost by providing cost estimates before sending API requests.
- 💾 Caching: Uses aiohttp_client_cach to avoid redundant requests and reduce cost by caching previous requests. Supports SQLite, MongoDB, DynamoDB and Redis cache backends.
- 📝 Request Logging: Implements Python's native logging framework for tracking and logging all API requests.

To get started, check the examples:
https://github.com/qagentur/texttunnel/tree/main/examples

OpenAI's function calling guide is also a useful resource:
https://platform.openai.com/docs/guides/gpt/function-calling

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Modules
=======

Chat Module
^^^^^^^^^^^
.. automodule:: texttunnel.chat
   :members:

Models Module
^^^^^^^^^^^^^
.. automodule:: texttunnel.models
   :members:

.. autoattribute:: texttunnel.models.GPT_4
.. autoattribute:: texttunnel.models.GPT_4_0613
.. autoattribute:: texttunnel.models.GPT_4_32K
.. autoattribute:: texttunnel.models.GPT_4_32K_0613
.. autoattribute:: texttunnel.models.GPT_4_0314
.. autoattribute:: texttunnel.models.GPT_4_32K_0314
.. autoattribute:: texttunnel.models.GPT_3_5_TURBO
.. autoattribute:: texttunnel.models.GPT_3_5_TURBO_16K
.. autoattribute:: texttunnel.models.GPT_3_5_TURBO_0613
.. autoattribute:: texttunnel.models.GPT_3_5_TURBO_16K_0613
.. autoattribute:: texttunnel.models.GPT_3_5_TURBO_0301

Models that are not included here can be created as custom instances of the Model class. Only chat models are supported; "instruct" models are not supported.

Preview models can be used, but will not be added as default models to the package. To use a preview model, create a custom instance of the Model class. Models that OpenAI deprecates will be removed from the package. This primarily affects date-versioned models.

Note that the model class attributes tokens_per_minute (TPM) and requests_per_minute (RPM) are based on tier 1 usage limits. See https://platform.openai.com/docs/guides/rate-limits?context=tier-free for more details. If your account has a higher usage tier, override the class attributes with your own values.

texttunnel does not track tokens_per_day (TPD) limits and assumes that it is the only process that is using your model quota.

Processor Module
^^^^^^^^^^^^^^^^
.. automodule:: texttunnel.processor
   :members:

Utils Module
^^^^^^^^^^^^
.. automodule:: texttunnel.utils
   :members:

Logging
=======

The package uses the standard logging library and creates a logger named "texttunnel".

To enable logging, add the following code to your script:

.. code-block:: python

   import logging
   logging.basicConfig(level=logging.WARNING) # choose whatever level you want
   logging.getLogger("texttunnel").setLevel(logging.INFO) # set to DEBUG for more verbose logging
    

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
