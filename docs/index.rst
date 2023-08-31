.. texttunnel documentation master file, created by
   sphinx-quickstart on Fri Aug 18 14:26:34 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to texttunnel's documentation!
======================================

This package offers a straightforward interface for integrating the GPT-3.5 and GPT-4 models into your natural language processing pipelines. It is optimally designed for the following scenario:

Suppose you possess a corpus of text data that you want to analyze using the GPT-3.5 or GPT-4 models. The goal is to perform extractive NLP tasks such as classification, named entity recognition, translation, summarization, question answering, or sentiment analysis. In this context, the package prioritizes efficiency and tidiness to provide you streamlined results.

Features:

- 📄 Output Schema: Utilizes JSON Schema alongside OpenAI's function calling schema to define the output data structure.
- ✔️ Input Validation: Ensures well-structured and error-free API requests by validating input data.
- ✅ Output Validation: Checks the response data from OpenAI's API against the expected schema to maintain data integrity.
- 🚀 Efficient Batching: Supports bulk processing by packing multiple input texts into a single request for the OpenAI's API.
- 🚦 Asynchronous Requests: Facilitates speedy data processing by sending simultaneous requests to OpenAI's API, while maintaining API rate limits.
- 💰 Cost Estimation: Aims for transparency in API utilization cost by providing cost estimates before sending API requests.
- 💾 Disk Caching: Uses diskcache to avoid redundant requests and reduce cost by caching previous requests.
- 📝 Request Logging: Implements Python's native logging framework for tracking and logging all API requests.

To get started, check the examples:
https://github.com/qagentur/texttunnel/tree/main/examples

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

Processor Module
^^^^^^^^^^^^^^^^
.. automodule:: texttunnel.processor
   :members:

Utils Module
^^^^^^^^^^^^
.. automodule:: texttunnel.utils
   :members:



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
