# Changelog for texttunnel

## 0.3.2

- `chat.build_requests` and `chat.build_binpacked_requests` now raise a ValueError when the text argument contains duplicates
- `aprocess_api_requests` now raises a ValueError when the requests passed to it have duplicate hashes

Both of these changes are to prevent waste of money on duplicate API requests. They also prevent a sorting error where results wouldn't be returned in the same order as the requests were passed in.

## 0.3.1

- Made `aprocess_api_requests()` independently useable to allow advanced users to take full control of the asyncio event loop.
- Added text classification example.

## 0.3.0

- Breaking: Replaced diskcache with aiohttp_client_cache for caching requests. This provides support for SQLite, Redis, DynamoDB and MongoDB cache backends.

## 0.2.3

- Added support for gpt-3.5-turbo-16k

## 0.2.2

- Initial release
