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

from dataclasses import (
    dataclass,
    field,
)
from pathlib import Path  # for saving results to a file
from typing import Any, Dict, Generator, List, Optional, Union  # for type hints

import aiohttp

# for storing API inputs, outputs, and metadata
import diskcache as dc  # for caching API responses

from texttunnel.chat import ChatCompletionRequest
from texttunnel.utils import hash_dict


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
    logging_level: int = 20,
    max_attempts: int = 10,
    rate_limit_headroom_factor: float = 0.75,
    api_key: Optional[str] = None,
    cache: Optional[dc.core.Cache] = None,
) -> List[Dict[str, Any]]:
    """

    Using the OpenAI API to process lots of text quickly takes some care.
    If you trickle in a million API requests one by one, they'll take days to complete.
    If you flood a million API requests in parallel, they'll exceed the rate limits
    and fail with errors. To maximize throughput, parallel requests need to be
    throttled to stay under rate limits.

    The following functions parallelizes requests to the OpenAI API while
    throttling to stay under rate limits.

    Features:
    - Streams requests from file, to avoid running out of memory for giant jobs
    - Makes requests concurrently, to maximize throughput
    - Throttles request and token usage, to stay under rate limits
    - Retries failed requests up to {max_attempts} times, to avoid missing data
    - Logs errors, to diagnose problems with requests

    Processes API requests in parallel, throttling to stay under rate limits.
    This function is a wrapper for aprocess_api_requests() that runs it in an asyncio
    event loop. Also sorts the output by request ID, so that the results are in
    the same order as the requests.

    Args:
        requests: List[ChatCompletionRequest]
            The requests to process, see ChatCompletionRequest class for details
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
        logging_level: int, optional
            level of logging to use; higher numbers will log fewer messages
            40 = ERROR; will log only when requests fail after all retries
            30 = WARNING; will log when requests his rate limits or other errors
            20 = INFO; will log when requests start and the status at finish
            10 = DEBUG; will log various things as the loop runs to see when they occur
            if omitted, will default to 20 (INFO).
        max_attempts: int, optional
            Number of times to retry a failed request before giving up
            if omitted, will default to 5
        rate_limit_headroom_factor: float, optional
            Factor to multiply the rate limit by to guarantee that the script
            stays under the limit if omitted, will default to 0.75
            (75% of the rate limit)
        api_key: str, optional
            API key to use
            if omitted, the function will attempt to read it from an environment
            variable {os.getenv("OPENAI_API_KEY")}
            If all requests are cached, the API key is not required.
        cache: diskcache.Cache, optional
            If provided, API responses will be served from the cache if available.
            New responses will be saved to the cache.
            Create a cache object with `diskcache.Cache("path/to/cache")`

    Returns:
        List[Dict[str, Any]]: list where each element consists of two dictionaries:
            - the original request
            - the API response
    """

    # This function was adapted from openai-cookbook

    # The function is structured as follows:
    #    - Initialize things
    #    - In process_api_requests loop:
    #        - Get next request if one is not already waiting for capacity
    #        - Update available token & request capacity
    #        - If enough capacity available, call API
    #        - The loop pauses if a rate limit error is hit
    #        - The loop breaks when no tasks remain

    # Handle Notebook environment
    if "ipykernel" in sys.modules:
        import nest_asyncio

        nest_asyncio.apply()

    # Handle save file
    output_filepath = prepare_output_filepath(output_filepath, keep_file)

    # Remember the order of the requests so that we can sort the results
    request_order = {request.get_hash(): i for i, request in enumerate(requests)}

    asyncio.run(
        aprocess_api_requests(
            requests=requests,
            output_filepath=output_filepath,
            logging_level=logging_level,
            max_attempts=max_attempts,
            rate_limit_headroom_factor=rate_limit_headroom_factor,
            api_key=api_key,
            cache=cache,
        )
    )

    # Read results from file
    with open(output_filepath, "r") as f:
        responses = [json.loads(line) for line in f]

    # Sort results in order of input requests
    # Results is a list of lists, where each sublist is [request, response]
    responses = sorted(
        responses,
        key=lambda x: request_order[hash_dict(x[0])],
    )

    assert len(responses) == len(requests)

    # Delete file if keep_file is False
    if not keep_file:
        output_filepath.unlink()
    else:
        # Overwrite file with sorted results
        with open(output_filepath, "w") as f:
            for response in responses:
                f.write(json.dumps(response) + "\n")

    return responses


async def aprocess_api_requests(
    requests: List[ChatCompletionRequest],
    output_filepath: Path,
    logging_level: int,
    max_attempts: int,
    rate_limit_headroom_factor: float,
    api_key: Optional[str] = None,
    cache: Optional[dc.Cache] = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # Check if requests can be served from the cache
    # Build a queue of requests that need to be sent to the API
    if cache is not None:
        logging.debug("Checking cache for requests.")
        requests_queue = []
        for request in requests:
            cache_key = request.get_hash()

            if cache_key in cache:
                response = cache[cache_key]

                data = [request.to_dict(), response]
                append_to_jsonl(data, output_filepath)
            else:
                requests_queue.append(request)

        request_cache_hits = len(requests) - len(requests_queue)
        logging.info(
            f"Found {request_cache_hits} out of {len(requests)} requests in cache."
        )
    else:
        logging.debug("No cache provided.")
        requests_queue = requests.copy()

    if len(requests_queue) == 0:
        return

    logging.debug("Cache check complete.")

    # Check that all requests use the same model. Otherwise, we can't set
    # a single rate limit for all requests.
    if len(set([request.model.name for request in requests])) > 1:
        raise ValueError("All requests must use the same model.")

    if rate_limit_headroom_factor < 0.01 or rate_limit_headroom_factor > 1:
        raise ValueError("rate_limit_headroom_factor must be between 0.01 and 1.")

    # initialize API constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    if api_key is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key is not None
        except AssertionError:
            raise ValueError(
                "api_key must be provided or set as an environment variable OPENAI_API_KEY."
            )

    request_url = "https://api.openai.com/v1/chat/completions"
    request_header = {"Authorization": f"Bearer {api_key}"}

    # initialize trackers
    queue_of_requests_to_retry = asyncio.Queue()
    task_id_generator = (
        task_id_generator_function()
    )  # generates integer IDs of 1, 2, 3, ...
    status_tracker = (
        StatusTracker()
    )  # single instance to track a collection of variables
    next_request = None  # variable to hold the next request to call

    # initialize available capacity counts
    max_requests_per_minute = (
        requests[0].model.requests_per_minute * rate_limit_headroom_factor
    )
    max_tokens_per_minute = (
        requests[0].model.tokens_per_minute * rate_limit_headroom_factor
    )

    available_request_capacity = max_requests_per_minute
    available_token_capacity = max_tokens_per_minute
    last_update_time = time.time()

    # initialize flags
    logging.debug("Initialization complete.")

    logging.info(
        f"Beginning main requests loop. {len(requests_queue)} requests to make."
    )

    # Main loop that runs until all tasks are finished
    while True:
        # get next request (if one is not already waiting for capacity)
        # check if there are requests that need to be retried
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(
                    f"Retrying request {next_request.task_id}: {next_request}"
                )
            # check if there are requests that haven't been tried yet
            if len(requests_queue) > 0:
                next_request = requests_queue.pop(0)

                # get new request
                next_request = APIRequest(
                    task_id=next(task_id_generator),
                    request=next_request,
                    token_consumption=next_request.count_total_tokens(),
                    attempts_left=max_attempts,
                )
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                logging.debug(f"Reading request {next_request.task_id}: {next_request}")

            else:
                logging.debug("All tasks have been started.")

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
            next_request_tokens = next_request.token_consumption
            if (
                available_request_capacity >= 1
                and available_token_capacity >= next_request_tokens
            ):
                # update counters
                available_request_capacity -= 1
                available_token_capacity -= next_request_tokens
                next_request.attempts_left -= 1

                # call API
                asyncio.create_task(
                    next_request.call_api(
                        request_url=request_url,
                        request_header=request_header,
                        retry_queue=queue_of_requests_to_retry,
                        output_filepath=output_filepath,
                        status_tracker=status_tracker,
                        cache=cache,
                    )
                )
                next_request = None  # reset next_request to empty

        # if all tasks are finished, break
        if status_tracker.num_tasks_in_progress == 0:
            break

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
            logging.warn(
                f"Pausing to cool down until {time.ctime(status_tracker.time_of_last_rate_limit_error + seconds_to_pause_after_rate_limit_error)}"
            )

    # after finishing, log final status
    logging.info("Parallel processing complete.")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {output_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
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
        cache: Optional[dc.Cache] = None,
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
            cache: A diskcache.Cache object to store API responses in. Optional.
        """

        error = None

        logging.info(f"Starting request #{self.task_id}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request.to_dict()
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = int(time.time())
                    status_tracker.num_rate_limit_errors += 1
                    status_tracker.num_api_errors -= (
                        1  # rate limit errors are counted separately
                    )

        except (
            Exception
        ) as e:  # catching naked exceptions is bad practice, but in this case we'll log & save them
            logging.warning(f"Request {self.task_id} failed with Exception {e}")
            status_tracker.num_other_errors += 1
            error = e

        if error:
            self.result.append(error)
            if self.attempts_left:
                retry_queue.put_nowait(self)
            else:
                logging.error(
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
            logging.debug(f"Request #{self.task_id} saved to {output_filepath}")

            # Store successful response in cache
            if cache is not None:
                cache_key = self.request.get_hash()
                cache[cache_key] = response
                logging.debug(f"Request #{self.task_id} cached")


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


def parse_response(response: List[Dict]) -> Dict[str, Any]:
    """
    Extract the function call arguments from a response.

    Args:
        response: The response to parse. It should be a list of length 2, where the
            first element is the request and the second element is the response.

    Returns:
        The function call arguments.
    """
    if len(response) != 2:
        raise ValueError(
            f"""
            Response {response} is incorrectly formatted.
            It should be a list of length 2, where the first element is the request
            and the second element is the response.
            """
        )

    try:
        return json.loads(
            response[1]["choices"][0]["message"]["function_call"]["arguments"]
        )
    except KeyError as e:
        raise ValueError(f"Response {response} is incorrectly formatted. Error: {e}")
    except json.decoder.JSONDecodeError as e:
        raise ValueError(f"Response {response} is not valid JSON. Error: {e}")


def parse_responses(responses: List[List[Dict]]) -> List[Dict[str, Any]]:
    """
    Extract the function call arguments from a list of responses.

    Args:
        responses: List of responses to parse.

    Returns:
        List where each element is the function call arguments for a response.
    """

    parsed = []
    for i, response in enumerate(responses):
        try:
            parsed.append(parse_response(response))
        except ValueError as e:
            raise ValueError(f"Response {i} is incorrectly formatted. Error: {e}")

    return parsed
