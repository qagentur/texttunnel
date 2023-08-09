# --------------------------------------------------------------------------------
# This file includes a function adapted from: openai-cookbook
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
import aiohttp  # for making API calls concurrently
import asyncio  # for running API calls concurrently
from hashlib import sha256  # for hashing API inputs
import json  # for saving results to a jsonl file
import logging  # for logging rate limit warnings and other messages
import os  # for reading API key from environment variable
import sys  # for checking notebook vs. script
import tiktoken  # for counting tokens
from typing import Any, Dict, Generator, List, Optional, Union  # for type hints
from pathlib import Path  # for saving results to a file
import time  # for sleeping after rate limit is hit
from dataclasses import (
    dataclass,
    field,
)  # for storing API inputs, outputs, and metadata

from texttunnel.chat import ChatCompletionRequest


def hash_dict(d: dict) -> str:
    """
    Hashes a dictionary using sha256.
    """
    return sha256(json.dumps(d).encode("utf-8")).hexdigest()


def process_api_requests(
    requests: List[ChatCompletionRequest],
    save_filepath: Union[str, Path],
    keep_file: bool = True,
    logging_level: int = 20,
    max_attempts: int = 10,
    rate_limit_headroom_factor: float = 0.75,
    token_encoding_name: str = "cl100k_base",
    api_key: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """

    Using the OpenAI API to process lots of text quickly takes some care.
    If you trickle in a million API requests one by one, they'll take days to complete.
    If you flood a million API requests in parallel, they'll exceed the rate limits and fail with errors.
    To maximize throughput, parallel requests need to be throttled to stay under rate limits.

    The following functions parallelizes requests to the OpenAI API while throttling to stay under rate limits.

    Features:
    - Streams requests from file, to avoid running out of memory for giant jobs
    - Makes requests concurrently, to maximize throughput
    - Throttles request and token usage, to stay under rate limits
    - Retries failed requests up to {max_attempts} times, to avoid missing data
    - Logs errors, to diagnose problems with requests

    Processes API requests in parallel, throttling to stay under rate limits.
    This function is a wrapper for aprocess_api_requests() that runs it in an asyncio event loop.
    Also sorts the output by request ID, so that the results are in the same order as the requests.

    Args:
        requests: List[ChatCompletionRequest]
            the requests to process, see ChatCompletionRequest class for details
        save_filepath: str, optional
            path to the file where the results will be saved
            file will be a jsonl file, where each line is an array with the original request plus the API response
            e.g., [{"model": "gpt-4", "messages": "..."}, {...}]
            if omitted, results will be saved to {requests_filename}_results.jsonl
        keep_file: bool, optional
            Whether to keep the results file after the script finishes.
        logging_level: int, optional
            level of logging to use; higher numbers will log fewer messages
            40 = ERROR; will log only when requests fail after all retries
            30 = WARNING; will log when requests his rate limits or other errors
            20 = INFO; will log when requests start and the status at finish
            10 = DEBUG; will log various things as the loop runs to see when they occur
            if omitted, will default to 20 (INFO).
        max_attempts: int, optional
            number of times to retry a failed request before giving up
            if omitted, will default to 5
        rate_limit_headroom_factor: float, optional
            factor to multiply the rate limit by to guarantee that the script stays under the limit
            if omitted, will default to 0.75 (75% of the rate limit)
        token_encoding_name: str, optional
            name of the token encoding used, as defined in the `tiktoken` package
            if omitted, will default to "cl100k_base" (used by GPT-3.5 and 4)
        api_key: str, optional
            API key to use
            if omitted, the function will attempt to read it from an environment variable {os.getenv("OPENAI_API_KEY")}

    Returns:
        List[Dict[str, Any]]: list where each element consists of two dictionaries:
            - the original request
            - the API response
    """

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

    if not isinstance(save_filepath, Path):
        save_filepath = Path(save_filepath)

    if save_filepath.exists():
        raise ValueError(f"File already exists: {save_filepath}")

    request_order = {
        hash_dict(request.to_dict()): i for i, request in enumerate(requests)
    }

    asyncio.run(
        aprocess_api_requests(
            requests,
            save_filepath,
            logging_level,
            max_attempts,
            rate_limit_headroom_factor,
            token_encoding_name,
            api_key,
        )
    )

    # Read results from file
    with open(save_filepath, "r") as f:
        responses = [json.loads(line) for line in f]

    # Sort results in order of input requests
    # Results is a list of lists, where each sublist is [request, response]
    responses = sorted(
        responses,
        key=lambda x: request_order[hash_dict(x[0])],
    )

    # Delete file if keep_file is False
    if not keep_file:
        save_filepath.unlink()
    else:
        # Overwrite file with sorted results
        with open(save_filepath, "w") as f:
            for response in responses:
                f.write(json.dumps(response) + "\n")

    return responses


async def aprocess_api_requests(
    requests: List[ChatCompletionRequest],
    save_filepath: Path,
    logging_level: int,
    max_attempts: int,
    rate_limit_headroom_factor: float,
    token_encoding_name: str,
    api_key: Optional[str] = None,
):
    """Processes API requests in parallel, throttling to stay under rate limits."""
    # constants
    seconds_to_pause_after_rate_limit_error = 15
    seconds_to_sleep_each_loop = (
        0.001  # 1 ms limits max throughput to 1,000 requests per second
    )

    # Check that all requests use the same model. Otherwise, we can't set
    # a single rate limit for all requests.
    if len(set([request.model.name for request in requests])) > 1:
        raise ValueError("All requests must use the same model.")

    if rate_limit_headroom_factor < 0.01 or rate_limit_headroom_factor > 1:
        raise ValueError("rate_limit_headroom_factor must be between 0.01 and 1.")

    if api_key is None:
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            assert api_key is not None
        except AssertionError:
            raise ValueError(
                "api_key must be provided or set as an environment variable OPENAI_API_KEY."
            )

    # initialize logging
    logging.basicConfig(level=logging_level)
    logging.debug(f"Logging initialized at level {logging_level}")

    # initialize API constants
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
    file_not_finished = True  # after file is empty, we'll skip reading it
    logging.debug("Initialization complete.")

    # initialize requests reading
    requests_queue = requests.copy()
    logging.debug("Requests loaded. Entering main loop.")

    while True:
        if len(requests_queue) > 0:
            next_request_input = requests_queue.pop(0)
        else:
            logging.debug("Requests queue exhausted")
            file_not_finished = False

        # get next request (if one is not already waiting for capacity)
        if next_request is None:
            if not queue_of_requests_to_retry.empty():
                next_request = queue_of_requests_to_retry.get_nowait()
                logging.debug(
                    f"Retrying request {next_request.task_id}: {next_request}"
                )
            elif file_not_finished:
                # get new request
                request_json = next_request_input.to_dict()
                next_request = APIRequest(
                    task_id=next(task_id_generator),
                    request_json=request_json,
                    token_consumption=num_tokens_consumed_from_request(
                        request_json, token_encoding_name
                    ),
                    attempts_left=max_attempts,
                    metadata=request_json.pop("metadata", None),
                )
                status_tracker.num_tasks_started += 1
                status_tracker.num_tasks_in_progress += 1
                logging.debug(f"Reading request {next_request.task_id}: {next_request}")

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
                        save_filepath=save_filepath,
                        status_tracker=status_tracker,
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
    logging.info(f"""Parallel processing complete. Results saved to {save_filepath}""")
    if status_tracker.num_tasks_failed > 0:
        logging.warning(
            f"{status_tracker.num_tasks_failed} / {status_tracker.num_tasks_started} requests failed. Errors logged to {save_filepath}."
        )
    if status_tracker.num_rate_limit_errors > 0:
        logging.warning(
            f"{status_tracker.num_rate_limit_errors} rate limit errors received. Consider running at a lower rate."
        )


# dataclasses
@dataclass
class StatusTracker:
    """Stores metadata about the script's progress. Only one instance is created."""

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
    """Stores an API request's inputs, outputs, and other metadata. Contains a method to make an API call."""

    task_id: int
    request_json: dict
    token_consumption: int
    attempts_left: int
    metadata: dict
    result: list = field(default_factory=list)

    async def call_api(
        self,
        request_url: str,
        request_header: dict,
        retry_queue: asyncio.Queue,
        save_filepath: Path,
        status_tracker: StatusTracker,
    ):
        """Calls the OpenAI API and saves results."""
        logging.info(f"Starting request #{self.task_id}")
        error = None
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=request_url, headers=request_header, json=self.request_json
                ) as response:
                    response = await response.json()
            if "error" in response:
                logging.warning(
                    f"Request {self.task_id} failed with error {response['error']}"
                )
                status_tracker.num_api_errors += 1
                error = response
                if "Rate limit" in response["error"].get("message", ""):
                    status_tracker.time_of_last_rate_limit_error = time.time()
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
                    f"Request {self.request_json} failed after all attempts. Saving errors: {self.result}"
                )
                data = (
                    [self.request_json, [str(e) for e in self.result], self.metadata]
                    if self.metadata
                    else [self.request_json, [str(e) for e in self.result]]
                )
                append_to_jsonl(data, save_filepath)
                status_tracker.num_tasks_in_progress -= 1
                status_tracker.num_tasks_failed += 1
        else:
            data = (
                [self.request_json, response, self.metadata]
                if self.metadata
                else [self.request_json, response]
            )
            append_to_jsonl(data, save_filepath)
            status_tracker.num_tasks_in_progress -= 1
            status_tracker.num_tasks_succeeded += 1
            logging.debug(f"Request {self.task_id} saved to {save_filepath}")


# functions
def append_to_jsonl(data: Any, filename: str) -> None:
    """
    Append a json payload to the end of a jsonl file.

    Args:
        data: The data to append.
        filename: The file to append to.
    """
    json_string = json.dumps(data)
    with open(filename, "a") as f:
        f.write(json_string + "\n")


def num_tokens_consumed_from_request(
    request_json: dict,
    token_encoding_name: str,
):
    """
    Count the number of tokens in the request.

    Args:
        request_json: The JSON payload of the request.
        token_encoding_name: The name of the token encoding to use.
    """
    encoding = tiktoken.get_encoding(token_encoding_name)

    # tokens = prompt + n * max_tokens
    max_tokens = request_json.get("max_tokens", 15)
    n = request_json.get("n", 1)
    completion_tokens = n * max_tokens

    num_tokens = 0
    for message in request_json["messages"]:
        num_tokens += (
            4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
        )
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":  # if there's a name, the role is omitted
                num_tokens -= 1  # role is always required and always 1 token
    num_tokens += 2  # every reply is primed with <im_start>assistant
    return num_tokens + completion_tokens


def task_id_generator_function() -> Generator[int, None, None]:
    """
    Generate integers 0, 1, 2, and so on.

    Returns:
        A generator that yields integers.
    """
    task_id = 0
    while True:
        yield task_id
        task_id += 1


def parse_response(response: List[Dict]) -> Dict[str, Any]:
    """
    Extract the function call arguments from a response.

    Args:
        response: The response to parse.

    Returns:
        The function call arguments.
    """
    return json.loads(
        response[1]["choices"][0]["message"]["function_call"]["arguments"]
    )


def parse_responses(responses: List[List[Dict]]) -> List[Dict[str, Any]]:
    """
    Extract the function call arguments from a list of responses.

    Args:
        responses: List of responses to parse.

    Returns:
        List where each element is the function call arguments for a response.
    """
    return [parse_response(r) for r in responses]
