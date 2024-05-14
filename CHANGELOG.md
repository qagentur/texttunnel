# Changelog for texttunnel

## 0.3.7

- Added model configurations for gpt-4-turbo and gpt-4o (note that users can add their own model configurations too)
- Added an example for named entity recognition

## 0.3.6

Bug fixes:

- Fixed a bug that caused retry requests to be overwritten by new requests in `aprocess_api_requests`.

## 0.3.5

Changes:

- texttunnel can now be used with any OpenAI Chat model, including your own fine-tuned models. Previously only a limited number of models were allowed due to peculiarities of token counting. This change comes at the cost of the possibility of miscounting tokens by 1 token per message in a chat, in case OpenAI changes token counting in future models. See https://github.com/qagentur/texttunnel/pull/70 for details.
- Requests now use a seed by default, which makes the results more consistent (see https://platform.openai.com/docs/guides/text-generation/reproducible-outputs).

Documentation:

- Documentation for changing API quota limits has been added to Sphinx docs.
- Documentation on texttunnel's model class support has been added to Sphinx docs.

## 0.3.4

Changes:

- additional DEBUG level logs for cached requests

Bug fixes:

- `aprocess_api_requests` no longer gets stuck after a request fails
- aiohttp sessions are now properly closed after an error occurs in the request

## 0.3.3

Changes:

- `aprocess_api_requests` now makes cache lookup asynchronously to improve performance
- the package is now compatible with jsonschema 3.0.0 and up, previously it was only compatible with 4.0.0 and up

Bug fixes:

- `aprocess_api_requests` now properly closes the connection to the cache backend

## 0.3.2

Changes:

- `chat.build_requests` and `chat.build_binpacked_requests` now raise a ValueError when the text argument contains duplicates
- `aprocess_api_requests` now raises a ValueError when the requests passed to it have duplicate hashes

Both of these changes are to prevent waste of money on duplicate API requests. They also prevent a sorting error where results wouldn't be returned in the same order as the requests were passed in.

Bug fixes:

- Fixed a bug where aiohttp sessions were not closed when an error occurred in the request

## 0.3.1

- Made `aprocess_api_requests()` independently useable to allow advanced users to take full control of the asyncio event loop.
- Added text classification example.

## 0.3.0

- Breaking: Replaced diskcache with aiohttp_client_cache for caching requests. This provides support for SQLite, Redis, DynamoDB and MongoDB cache backends.

## 0.2.3

- Added support for gpt-3.5-turbo-16k

## 0.2.2

- Initial release
