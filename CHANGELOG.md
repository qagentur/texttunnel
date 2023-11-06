# Changelog for texttunnel

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
