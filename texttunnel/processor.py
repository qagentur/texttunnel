# --------------------------------------------------------------------------------
# This file includes classes and functions adapted from: openai-cookbook
# Original source code: https://github.com/openai/openai-cookbook/blob/c651bfdda64ac049747c2a174cde1c946e2baf1d/examples/api_request_parallel_processor.py
# Copyright (c) 2023 OpenAI

# MIT License

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# imports
import asyncio  # for running API calls concurrently
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key from environment variable
import sys  # for checking notebook vs. script
import tempfile  # for creating a temporary file to save results
import time  # for sleeping after rate limit is hit
from dataclasses import dataclass, field
from pathlib import Path  # for saving results to a file
from typing import Any, Dict, Generator, List, Optional, Union  # for type hints

import aiohttp
import aiohttp_client_cache

# for storing API inputs, outputs, and metadata
import jsonschema  # for validating API responses

from texttunnel.chat import ChatCompletionRequest
from texttunnel.models import Model
from texttunnel.utils import hash_dict

Response = List[Dict[str, Any]]

logger = logging.getLogger("texttunnel")


def prepare_output_filepath(
    output_filepath: Optional[Union[str, Path]], keep_file: bool
) -> Path:
    """
    Validates the output_filepath and returns a Path object. Uses a temporary file
    if output_filepath is None.

    Args:
        output_filepath: The path to save the results to. If None, a temporary file
            will be used.
        keep_file: Whether to keep the file after the function returns. If True,
            output_filepath must not be None.

    Returns:
        A Path object representing the output_filepath.
    """
    using_tempfile = False

    if output_filepath is None:
        output_filepath = tempfile.NamedTemporaryFile(delete=False).name
        using_tempfile = True
        if keep_file:
            raise ValueError(
                "keep_file=True is not compatible with output_filepath=None"
            )

    if not isinstance(output_filepath, Path):
        output_filepath = Path(output_filepath)

    if output_filepath.exists() and not using_tempfile:
        raise ValueError(f"File already exists: {output_filepath}")

    return output_filepath


def process_api_requests(
    requests: List[ChatCompletionRequest],
    output_filepath: Optional[Union[str, Path]] = None,
    keep_file: bool = False,
    max_attempts: int = 10,
    rate_limit_headroom_factor: float = 0.75,
    api_key: Optional[str] = None,
    cache: Optional[aiohttp_client_cache.CacheBackend] = None,
) -> List[Response]:
    """
    Make requests to OpenAI. This function is a wrapper around
    aprocess_api_requests() that executes it within asyncio.run, saving you the
    trouble of having to use asyncio directly.

    Note that if you're running this function in a Jupyter notebook, the function
    will automatically import nest_asyncio and call nest_asyncio.apply() to allow
    a second event loop to run in the same process. This is necessary because
    Jupyter notebooks already run an event loop in the background.

    If you require more control over the event loop, use the coroutine
    aprocess_api_requests() instead.

    Args:
        requests: List[ChatCompletionRequest]
            The requests to process, see ChatCompletionRequest class for details.
            Duplicate requests are not allowed.
        output_filepath: str, optional
            Path to the file where the results will be saved
            file will be a jsonl file, where each line is an array with the original
            request plus the API response e.g.,
            [{"model": "gpt-4", "messages": "..."}, {...}]
            if omitted, the results will be saved to a temporary file.
        keep_file: bool, optional
            Whether to keep the results file after the script finishes, in addition
            to the results being returned by the function.
            Defaults to False, so the file will be deleted after the script finishes.
            Setting this to True is not compatible with output_filepath=None.
        max_attempts: int, optional
            Number of times to retry a failed request before giving up
            if omitted, will default to 5
        rate_limit_headroom_factor: float, optional
            Factor to multiply the rate limit by to guarantee that the script
            stays under the limit if omitted, will default to 0.75
            (75% of the rate limit)
        api_key: str, optional
            API key to use. If omitted, the function will attempt to read it
            from an environment variable OPENAI_API_KEY. If that fails, an error
            will be raised, unless all requests are cached.
        cache: aiohttp_client_cache.CacheBackend, optional
            If provided, API responses will be served from the cache if available.
            New responses will be saved to the cache.
            Check the aiohttp_client_cache documentation for details on the
            available cache backends and how to configure them. See
            https://aiohttp-client-cache.readthedocs.io/en/stable/backends.html.
            Each backend requires different dependencies. For example, the SQLite
            backend requires the package "aiosqlite" to be installed.

    Returns:
        List[Dict[str, Any]]: list where each element consists of two dictionaries:
            - the original request
            - the API response
    """

    # Handle Notebook environment
    if "ipykernel" in sys.modules:
        # nest_asyncio is a workaround for running asyncio in Jupyter notebooks
        # it's always available when ipykernel is available
        import nest_asyncio

        nest_asyncio.apply()
        logger.info(
            "Running in Jupyter notebook environment. Activated nest_asyncio to allow asyncio to run."
        )

    responses = asyncio.run(
        aprocess_api_requests(
            requests=requests,
            output_filepath=output_filepath,
            keep_file=keep_file,
            max_attempts=max_attempts,
            rate_limit_headroom_factor=rate_limit_headroom_factor,
            api_key=api_key,
            cache=cache,
        )
    )

    return responses


async def fetch_json_response_from_cache(
    cache: aiohttp_client_cache.CacheBackend, url: str, request_json: dict
) -> Optional[dict]:
    """
    Fetch a response from the cache if it exists.

    Args:
        cache: Cache to fetch from.
        url: URL that was requested.
        request_json: JSON payload that was sent with the request.

    Returns:
        The cached response JSON if it exists, otherwise None.
    """
    cache_return_tuple = await cache.request(
        method="POST",  # ChatCompletion always uses POST requests
        url=url,
        json=request_json,
    )

    if cache_return_tuple[0] is None:
        return None

    cache_response_json = await cache_return_tuple[0].json()

    return cache_response_json


async def aprocess_api_requests(
    requests: List[ChatCompletionRequest],
    output_filepath: Optional[Union[str, Path]] = None,
    keep_file: bool = False,
    max_attempts: int = 10,
    rate_limit_headroom_factor: float = 0.75,
    api_key: Optional[str] = None,
    cache: Optional[aiohttp_client_cache.CacheBackend] = None,
) -> List[Response]:
    """
    Make asynchronous requests to the OpenAI API while
    throttling to stay under rate limits.

    Features:
    - Makes requests concurrently, to maximize throughput
    - Throttles request and token usage, to stay under rate limits
    - Retries failed requests up to {max_attempts} times, to avoid missing data
    - Logs errors, to diagnose problems with requests


    Args:
        requests: List[ChatCompletionRequest]
            The requests to process, see ChatCompletionRequest class for details.
            Duplicate requests are not allowed.
        output_filepath: str, optional
            Path to the file where the results will be saved
            file will be a jsonl file, where each line is an array with the original
            request plus the API response e.g.,
            [{"model": "gpt-4", "messages": "..."}, {...}]
            if omitted, the results will be saved to a temporary file.
        keep_file: bool, optional
            Whether to keep the results file after the script finishes, in addition
            to the results being returned by the function.
            Defaults to False, so the file will be deleted after the script finishes.
            Setting this to True is not compatible with output_filepath=None.
        max_attempts: int, optional
            Number of times to retry a failed request before giving up
            if omitted, defaults to 5.
        rate_limit_headroom_factor: float, optional
            Factor to multiply the rate limit by to guarantee that the script
            stays under the limit if omitted, defaults to 0.75.
            (75% of the rate limit).
        api_key: str, optional
            API key to use. If omitted, the function will attempt to read it
            from an environment variable OPENAI_API_KEY. If that fails, an error
            will be raised, unless all requests are cached.
        cache: aiohttp_client_cache.CacheBackend, optional
            If provided, API responses will be served from the cache if available.
            New responses will be saved to the cache.
            Check the aiohttp_client_cache documentation for details on the
            available cache backends and how to configure them. See
            https://aiohttp-client-cache.readthedocs.io/en/stable/backends.html.
            Each backend has different dependencies. For example, the SQLite
            backend requires the package "aiosqlite" to be installed.

    Returns:
        List[Dict[str, Any]]: list where each element consists of two dictionaries:
            - the original request
            - the API response
    """

    if len(requests) != len(set([request.get_hash() for request in requests])):
        # Duplicate requests can cause problems with ordering of results
        # Plus it's probably a mistake and would waste money
        raise ValueError("Duplicate requests detected. Each request must be unique.")

    # This function was adapted from openai-cookbook

    # The function is structured as follows:
    #    - Initialize things
    #    - In API processing loop
    #        - Get next request if one is not already waiting for capacity
    #        - Update available token & request capacity
    #        - If enough capacity available, call API. Responses are written to file
    #        - The loop pauses if a rate limit error is hit
    #        - The loop breaks when no tasks remain
    #   - Fetch results from file
    #   - Sort results in order of input requests
    #   - Return results

    output_filepath = prepare_output_filepath(output_filepath, keep_file)

    # Remember the order of the requests so that we can sort the results
    # Duplicate requests are not allowed, so the hash of each request is unique
    request_order = {request.get_hash(): i for i, request in enumerate(requests)}

    request_url = "https://api.openai.com/v1/chat/completions"

    if cache:
        check_cache_settings(cache)

        # Check if requests can be served from the cache
        # Build a list of requests that need to be sent to the API
        # Handling cached requests separately allows us to avoid allocating
        # rate limit capacity to them and provide clearer logging.
        logger.debug("Checking cache for requests.")

        # Make asynchronous calls to the cache
        tasks = [
            fetch_json_response_from_cache(
                cache=cache,
                url=request_url,
                request_json=request.to_dict(),
            )
            for request in requests
        ]

        logger.debug("Created cache request tasks.")

        cached_responses = await asyncio.gather(*tasks)

        logger.debug("Gathered cached responses.")

        # Create a list of requests that need to be sent to the API
        requests_queue = []

        # Check cache responses, and add to queue if not found
        for request, response in zip(requests, cached_responses):
            if response is not None:
                # Add to results file
                data = [request.to_dict(), response]
                append_to_jsonl(data, output_filepath)
            else:
                requests_queue.append(request)

        request_cache_hits = len(requests) - len(requests_queue)
        logger.info(
            f"Found {request_cache_hits} out of {len(requests)} requests in cache."
        )
    else:
        logger.debug("No cache provided.")
        requests_queue = requests.copy()

    logger.debug("Cache check complete.")

    if len(requests_queue) > 0:
        await run_request_loop(
            requests_queue=requests_queue,
            request_url=request_url,
            output_filepath=output_filepath,
            cache=cache,
            max_attempts=max_attempts,
            rate_limit_headroom_factor=rate_limit_headroom_factor,
            api_key=api_key,
        )

    if cache:
        await cache.close()

    with open(output_filepath, "r") as f:
        request_response_pairs = [json.loads(line) for line in f]

    # Sort results in order of input requests
    # Results is a list of lists, where each sublist is [request, response]
    request_response_pairs = sorted(
        request_response_pairs,
        key=lambda x: request_order[hash_dict(x[0])],
    )

    assert len(request_response_pairs) == len(requests)

    if not keep_file:
        output_filepath.unlink()
    else:
        # Overwrite file with sorted results
        with open(output_filepath, "w") as f:
            for r in request_response_pairs:
                f.write(json.dumps(r) + "\n")

    return request_response_pairs


async def run_request_loop(
    requests_queue: List[ChatCompletionRequest],
    request_url: str,
    output_filepath: Path,
    cache: Optional[aiohttp_client_cache.CacheBackend] = None,
    max_attempts: int = 10,
    rate_limit_headroom_factor: float = 0.75,
    api_key: Optional[str] = None,
):
    """
    Run the main loop that processes API requests. Save results to a file.

    Args:
        requests_queue: A queue of requests to process.
        request_url: The URL to send the requests to.
        output_filepath: The path to the file where the results will be saved.
        cache: A aiohttp_client_cache.CacheBackend object that stores API
            responses. If provided, the response will be stored in the cache.
        max_attempts: Number of times to retry a failed request before giving up.
        rate_limit_headroom_factor: Factor to multiply the rate limit by to
            guarantee that the script stays under the limit.
        api_key: API key to use. If omitted, the function will attempt to read it
            from an environment variable OPENAI_API_KEY. If that fails, an error
            will be raised, unless all requests are cached.

    """

    # Check that all requests use the same model. Otherwise, we can't set
    # a single rate limit for all requests.
    if len(set([request.model.name for request in requests_queue])) > 1:
        raise ValueError("All requests must use the same model.")

    if rate_limit_headroom_factor < 0.01 or rate_limit_headroom_factor > 1:
        raise ValueError("rate_limit_headroom_factor must be between 0.01 and 1.")

    # initialize API constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # initialize API authentication
    if api_key is None:
        api_key = fetch_api_key()

    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    retry_queue = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    max_requests_per_minute = (
        requests_queue[0].model.requests_per_minute * rate_limit_headroom_factor
    )
    max_tokens_per_minute = (
        requests_queue[0].model.tokens_per_minute * rate_limit_headroom_factor
    )

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    logger.debug("Initialization complete.")

    logger.info(
        f"Beginning main requests loop. {len(requests_queue)} requests to make."
    )

    # Main loop that runs until all tasks are finished
    last_status_log_timestamp = time.time()

    while True:
        # get next request if one is not already waiting for capacity
        if next_request is None:
            # retry a request if one is waiting in the retry queue
            if not retry_queue.empty():
                next_request = retry_queue.get_nowait()
                logger.debug(f"Retrying request {next_request.task_id}: {next_request}")

            # send a new request if one is waiting in the requests queue
            elif len(requests_queue) > 0:
                next_chat_completion = requests_queue.pop(0)

                # get new request
                next_request = APIRequest(
                    task_id=next(task_id_generator),
                    request=next_chat_completion,
                    token_consumption=next_chat_completion.count_total_tokens(),
                    attempts_left=max_attempts,
                )
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                logger.debug(f"Reading request {next_request.task_id}: {next_request}")

        # update available capacity
        current_time = time.time()
        seconds_since_update = current_time - last_update_time
        available_request_capacity = min(
            available_request_capacity
            + max_requests_per_minute * seconds_since_update / 60.0,
            max_requests_per_minute,
        )
        available_token_capacity = min(
            available_token_capacity
            + max_tokens_per_minute * seconds_since_update / 60.0,
            max_tokens_per_minute,
        )
        last_update_time = current_time

        # if enough capacity available, call API
        if next_request:
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request.token_consumption
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request.token_consumption
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=retry_queue,
                        output_filepath=output_filepath,
                        status_tracker=status_tracker,
                        cache=cache,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break
        else:
            # Log status every 10 seconds
            if time.time() - last_status_log_timestamp > 10:
                logger.debug(
                    "%s tasks in progress. Successful tasks: %s. Failed tasks: %s. "
                    "Rate limit errors: %s. Other errors: %s. Retry queue length: %s. "
                    "Tasks not yet tried: %s. ",
                    status_tracker.num_tasks_in_progress,
                    status_tracker.num_tasks_succeeded,
                    status_tracker.num_tasks_failed,
                    status_tracker.num_rate_limit_errors,
                    status_tracker.num_other_errors,
                    retry_queue.qsize(),
                    len(requests_queue),
                )
                last_status_log_timestamp = time.time()

        # main loop sleeps briefly so concurrent tasks can run
        await asyncio.sleep(seconds_to_sleep_each_loop)

        # if a rate limit error was hit recently, pause to cool down
        seconds_since_rate_limit_error = (
            time.time() - status_tracker.time_of_last_rate_limit_error
        )
        if seconds_since_rate_limit_error < seconds_to_pause_after_rate_limit_error:
            remaining_seconds_to_pause = (
                seconds_to_pause_after_rate_limit_error - seconds_since_rate_limit_error
            )
            await asyncio.sleep(remaining_seconds_to_pause)
            # ^e.g., if pause is 15 seconds and final limit was hit 5 seconds ago
            logger.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    logger.info("Parallel processing complete.")
    if status_tracker.num_tasks_failed > 0:
        logger.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {output_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logger.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )


def check_cache_settings(cache: aiohttp_client_cache.CacheBackend) -> None:
    """
    Check that the cache is configured correctly to work with texttunnel.
    Raises a ValueError if the cache is not configured correctly.

    Args:
        cache: The cache to check.
    """
    if "POST" not in cache.allowed_methods:
        raise ValueError(
            'cache.allowed_methods must include "POST". Add the argument "allowed_methods=["POST"]" to the cache constructor.'
        )

    if cache.include_headers:
        raise ValueError("cache.include_headers must be False to protect the API key.")


def fetch_api_key() -> str:
    """
    Fetch the API key from the environment variable OPENAI_API_KEY. Raises a
    ValueError if the API key is not found.

    Returns:
        The API key.
    """

    try:
        api_key = os.getenv("OPENAI_API_KEY")
        assert api_key is not None
        return api_key
    except AssertionError:
        raise ValueError(
            "OPENAI_API_KEY environment variable not found. Please set it and try again."
        )


# dataclasses
@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

    # This class was adapted from openai-cookbook

    num_tasks_started: int = 0
    num_tasks_in_progress: int = 0  # script ends when this reaches 0
    num_tasks_succeeded: int = 0
    num_tasks_failed: int = 0
    num_rate_limit_errors: int = 0
    num_api_errors: int = 0  # excluding rate limit errors, counted above
    num_other_errors: int = 0
    time_of_last_rate_limit_error: int = 0  # used to cool off after hitting rate limits


@dataclass
class APIRequest:
    """Stores an API request's inputs, outputs, and other metadata.
    Contains a method to make an API call."""

    task_id: int
    request: ChatCompletionRequest
    token_consumption: int
    attempts_left: int
    result: list = field(default_factory=list)

    # This class was adapted from openai-cookbook

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        output_filepath: Path,
        status_tracker: StatusTracker,
        cache: Optional[aiohttp_client_cache.CacheBackend] = None,
        timeout_seconds: int = 120,
    ):
        """
        Calls the OpenAI API and appends the request and result to a JSONL file.
        If a cache provided, the result will be stored in the cache.
        The cache key is the hash of the request.

        Args:
            request_url: The URL to send the request to.
            request_header: The header to send with the request.
            retry_queue: A queue of requests that need to be retried.
                Will be populated if the request fails.
            output_filepath: The path to the file where the results will be saved.
            status_tracker: A StatusTracker object that tracks the greater
                request loop's progress.
            cache: A aiohttp_client_cache.CacheBackend object that stores API
                responses. If provided, the response will be stored in the cache.
            timeout_seconds: The number of seconds to wait for a response before
                timing out. Defaults to 120 seconds.
        """

        error = None

        logger.info(f"Starting request #{self.task_id}")
        timeout = aiohttp.ClientTimeout(total=timeout_seconds)

        # Choose the session class based on whether cache is provided
        session_class = (
            aiohttp.ClientSession
            if cache is None
            else aiohttp_client_cache.CachedSession
        )

        session_kwargs = (
            {"timeout": timeout}
            if cache is None
            else {"cache": cache, "timeout": timeout}
        )

        try:
            async with session_class(**session_kwargs) as session:
                async with session.post(
                    url=request_url,
                    headers=request_header,
                    json=self.request.to_dict(),
                ) as response:
                    response = await response.json()

                if "error" in response:
                    # API and rate limit errors don't raise an exception
                    # They are found in the response JSON
                    logger.warning(
                        f"Request {self.task_id} failed with error {response['error']}"
                    )

                    error = response
                    if "Rate limit" in response["error"].get("message", ""):
                        status_tracker.time_of_last_rate_limit_error = int(time.time())
                        status_tracker.num_rate_limit_errors += 1
                    else:
                        status_tracker.num_api_errors += 1

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logger.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
                logger.debug(
                    "Added request #%s to retry queue. Queue length: %s.",
                    self.task_id,
                    retry_queue.qsize(),
                )
            else:
                logger.error(
                    f"Request {self.request.to_dict()} failed after all attempts. Saving errors: {self.result}"
                )
                data = [self.request.to_dict(), [str(e) for e in self.result]]
                append_to_jsonl(data, output_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:  # success
            data = [self.request.to_dict(), response]
            append_to_jsonl(data, output_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logger.debug(f"Request #{self.task_id} saved to {output_filepath}")


# functions
def append_to_jsonl(data: Any, filename: Path) -> None:
    """
    Append a json payload to the end of a jsonl file.

    Args:
        data: The data to append.
        filename: The file to append to.
    """
    # This function was adapted from openai-cookbook

    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def task_id_generator_function() -> Generator[int, None, None]:
    """
    Generate integers 0, 1, 2, and so on.

    Returns:
        A generator that yields integers.
    """
    # This function was adapted from openai-cookbook

    task_id = 0
    while True:
        yield task_id
        task_id += 1


RESPONSE_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "array",
    "items": [
        # Request schema
        {
            "type": "object",
            "properties": {
                "model": {"type": "string"},
                "max_tokens": {"type": "integer"},
                "messages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "role": {"type": "string"},
                            "content": {"type": "string"},
                        },
                        "required": ["role", "content"],
                    },
                },
                "functions": {
                    "type": "array",
                    "properties": {
                        "name": {"type": "string"},
                        "parameters": {"type": "object"},
                    },
                    "required": ["name", "parameters"],
                },
            },
            "required": [
                "model",
                "max_tokens",
                "messages",
                "functions",
            ],
        },
        # Response schema
        {
            "type": "object",
            "properties": {
                "id": {"type": "string"},
                "object": {"type": "string"},
                "created": {"type": "integer"},
                "model": {"type": "string"},
                "choices": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "index": {"type": "integer"},
                            "message": {
                                "type": "object",
                                "properties": {
                                    "role": {"type": "string"},
                                    "content": {"type": ["string", "null"]},
                                    "function_call": {
                                        "type": "object",
                                        "properties": {
                                            "name": {"type": "string"},
                                            "arguments": {"type": "string"},
                                        },
                                        "required": ["name", "arguments"],
                                    },
                                },
                                "required": ["role", "function_call"],
                            },
                            "finish_reason": {"type": "string"},
                        },
                        "required": ["index", "message", "finish_reason"],
                    },
                },
                "usage": {
                    "type": "object",
                    "properties": {
                        "prompt_tokens": {"type": "integer"},
                        "completion_tokens": {"type": "integer"},
                        "total_tokens": {"type": "integer"},
                    },
                    "required": ["prompt_tokens", "completion_tokens", "total_tokens"],
                },
            },
            "required": ["id", "object", "created", "model", "choices", "usage"],
        },
    ],
}


def is_valid_response(response: Response, print_errors=False) -> bool:
    """
    Check if a response conforms to the response JSON schema.
    """
    try:
        jsonschema.validate(response, RESPONSE_SCHEMA)
        return True
    except jsonschema.exceptions.ValidationError as e:
        if print_errors:
            print(e)
        return False


def parse_arguments(response: Response) -> Dict[str, Any]:
    """
    Extract the function call arguments from a response.

    Args:
        response: The response to parse. It should be a list of length 2, where the
            first element is the request and the second element is the response.

    Returns:
        The function call arguments.
    """

    if not is_valid_response(response):
        raise ValueError("Response is not valid.")

    return json.loads(
        response[1]["choices"][0]["message"]["function_call"]["arguments"]
    )


def parse_token_usage(response: Response) -> Dict[str, Any]:
    """
    Extract the token usage from a response.

    Args:
        response: The response to parse. It should be a list of length 2, where the
            first element is the request and the second element is the response.

    Returns:
        The token usage.
    """
    if not is_valid_response(response):
        raise ValueError("Response is not valid.")

    return response[1]["usage"]


def usage_to_cost(usage: Dict, model: Model):
    """
    Convert token usage to cost in USD.

    Args:
        usage: The token usage. Retrieve it with parse_token_usage().
        model: The model used to generate the response.

    Returns:
        The cost in USD.
    """
    input_cost = model.input_token_price_per_1k * usage["prompt_tokens"] / 1000
    output_cost = model.output_token_price_per_1k * usage["completion_tokens"] / 1000
    total_cost = input_cost + output_cost
    return total_cost
