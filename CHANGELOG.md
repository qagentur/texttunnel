# Changelog for texttunnel

## 0.3.1

- Made `aprocess_api_requests()` independently useable to allow advanced users to take full control of the asyncio event loop.
- Added text classification example.

## 0.3.0

- Breaking: Replaced diskcache with aiohttp_client_cache for caching requests. This provides support for SQLite, Redis, DynamoDB and MongoDB cache backends.

## 0.2.3

- Added support for gpt-3.5-turbo-16k

## 0.2.2

- Initial release