# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from threading import Lock
from typing import List, Optional, Tuple, Union

import tensorrt_llm.bindings.executor as trtllm
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from vllm.lora.request import LoRARequest
from vllm.outputs import CompletionOutput, RequestOutput
from vllm.sampling_params import SamplingParams
from vllm.sequence import Logprob, SampleLogprobs, SequenceStatus
from vllm.transformers_utils.detokenizer import (
    INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET,
    _convert_tokens_to_string_with_added_encoders,
)
from vllm_mxext.trtllm.utils import to_trt_req


class TrtRequest:
    def __init__(
        self,
        req_id: str,
        prompt: str,
        prompt_ids: List[int],
        max_model_len: int,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        sampling_params: SamplingParams,
        lora_request: None | LoRARequest = None,
        trt_req_id: int = 0,
        arrival_time: float = 0,
    ) -> None:
        self._req_id = req_id
        self._trt_req_id = trt_req_id
        self._prompt: str = prompt
        self._prompt_ids: List[int] = prompt_ids
        self._gen_ids: List[int] = []
        self._tokens = None
        self._generation_str = ""
        self._prefix_offset = 0
        self._read_offset = 0
        self._error = None
        self._max_model_len = max_model_len
        self._tokenizer = tokenizer
        self._sampling_params = sampling_params
        self._lora_request = lora_request
        self._seq_status = SequenceStatus.RUNNING
        self._seq_stop_reason = None
        self._new_char_count = 0
        self._arrival_time = arrival_time
        self._last_token_time = 0
        self._lora_cache_miss_count = 0
        self._output_logprobs: List[float] = []
        self._cum_logprob: float = 0.0

        self._lock = Lock()

    @property
    def request_id(self) -> str:
        return self._req_id

    @property
    def trt_request_id(self) -> int:
        return self._trt_req_id

    @property
    def lora_cache_miss_count(self) -> int:
        return self._lora_cache_miss_count

    def record_lora_cache_miss(self) -> None:
        self._lora_cache_miss_count += 1

    @property
    def lora_request(self):
        return self._lora_request

    @lora_request.setter
    def lora_request(self, lora_request: LoRARequest):
        self.lora_request = lora_request

    @trt_request_id.setter
    def trt_request_id(self, req_id: int):
        self._trt_req_id = req_id

    @property
    def arrival_time(self):
        return self._arrival_time

    @property
    def last_token_time(self):
        return self._last_token_time

    @last_token_time.setter
    def last_token_time(self, value):
        self._last_token_time = value

    @property
    def seq_status(self):
        return self._seq_status

    @property
    def num_prompt_tokens(self):
        return len(self._prompt_ids)

    @property
    def num_generated_tokens(self):
        return len(self._tokens) if self._tokens is not None else 0

    def __eq__(self, o) -> bool:
        return (
            self._req_id == o._req_id
            and self._trt_req_id == o._trt_req_id
            and self._prompt == o._prompt
            and self._prompt_ids == o._prompt_ids
            and self._gen_ids == o._gen_ids
            and self._tokens == o._tokens
            and self._generation_str == o._generation_str
            and self._prefix_offset == o._prefix_offset
            and self._read_offset == o._read_offset
            and self._error == o._error
            and self._max_model_len == o._max_model_len
            and self._tokenizer == o._tokenizer
            and self._sampling_params == o._sampling_params
            and self._lora_request == o._lora_request
            and self._seq_status == o._seq_status
            and self._new_char_count == o._new_char_count
        )

    def finished(self) -> bool:
        with self._lock:
            return SequenceStatus.is_finished(self._seq_status)

    def _set_seq_status(self, status: SequenceStatus):
        self._seq_status = status

    def postprocess(self, response: trtllm.Response) -> Tuple[RequestOutput, int]:
        should_detokenize = self._add_response(response)
        num_new_tokens = 0
        if should_detokenize:
            num_new_tokens = self._detokenize()
            self._maybe_stop()
        return self.to_request_output(), num_new_tokens

    def _add_response(self, response: trtllm.Response) -> bool:
        should_detokenize = True
        if response.has_error():
            self._error = response.error_msg
            self._seq_status = SequenceStatus.FINISHED_ABORTED
            should_detokenize = False
        else:
            result = response.result
            # TODO (grclark) handle beam_size > 1 at some point
            # Very rarely trt-llm will output a negative token
            ## only seem when random weights are used, but we'll filter just to be safe
            new_tokens = list(filter(lambda x: x >= 0, result.output_token_ids[0]))
            if len(new_tokens) == 0:
                should_detokenize = False
            self._gen_ids.extend(new_tokens)
            # TODO (grclark) handle beam_size > 1 at some point
            if result.log_probs is not None:
                logprobs = result.log_probs[0]
                self._output_logprobs.extend(logprobs)
            if result.cum_log_probs is not None:
                self._cum_logprob = result.cum_log_probs[0]
            if result.is_final and len(self._gen_ids) >= self._sampling_params.max_tokens:
                self._seq_status = SequenceStatus.FINISHED_LENGTH_CAPPED
            elif result.is_final:
                self._seq_status = SequenceStatus.FINISHED_STOPPED
        return should_detokenize

    def _detokenize(self) -> int:
        new_toks, new_text, prefix_offset, read_offset = detokenize_incrementally(
            self._tokenizer,
            self._gen_ids,
            self._tokens,
            self._prefix_offset,
            self._read_offset,
            self._sampling_params.skip_special_tokens,
            self._sampling_params.spaces_between_special_tokens,
        )
        if self._tokens is None:
            self._tokens = []
        self._tokens.extend(new_toks)
        self._generation_str += new_text
        self._prefix_offset = prefix_offset
        self._read_offset = read_offset
        self._new_char_count = len(new_text)
        return len(new_toks)

    def _maybe_stop(self):
        if self.finished():
            return

        if self._sampling_params.ignore_eos:
            return

        if self._gen_ids[-1] == self._sampling_params.eos_token_id:
            self._seq_status = SequenceStatus.FINISHED_STOPPED
            return

        # Check if a stop token was encountered.
        # This assumes a single token produced per step.
        last_token_id = self._gen_ids[-1]
        if last_token_id in self._sampling_params.stop_token_ids:
            if self._new_char_count and (not self._sampling_params.include_stop_str_in_output):
                # Remove last token
                self._generation_str = self._generation_str[: -self._new_char_count]
            self._seq_status = SequenceStatus.FINISHED_STOPPED
            self._seq_stop_reason = last_token_id
            return

        stop_str = self._check_stops()
        if stop_str is not None:
            self._seq_status = SequenceStatus.FINISHED_STOPPED
            self._seq_stop_reason = stop_str
            return

    def to_request_output(self) -> RequestOutput:
        vllm_logprobs: SampleLogprobs | None = None
        if self._sampling_params.logprobs:
            vllm_logprobs = []
            num_tokens = len(self._gen_ids)
            try:
                for idx in range(num_tokens):
                    tok = self._tokens[idx] if self._tokens is not None and idx < len(self._tokens) else None
                    vllm_logprobs.append({self._gen_ids[idx]: Logprob(self._output_logprobs[idx], 1, tok)})
            except (ValueError, IndexError):
                raise Exception("_gen_ids, output_logprobs and tokens but be the same size. this should not happen")
        finish_reason = SequenceStatus.get_finished_reason(self._seq_status)
        output = RequestOutput(
            request_id=self.request_id,
            prompt=self._prompt,
            prompt_token_ids=self._prompt_ids,
            prompt_logprobs=None,
            outputs=[
                CompletionOutput(
                    0,
                    self._generation_str,
                    list(self._gen_ids),
                    self._cum_logprob,
                    vllm_logprobs,
                    finish_reason,
                    self._seq_stop_reason,
                )
            ],
            finished=self.finished(),
            metrics=None,
            lora_request=self._lora_request,
        )
        return output

    def to_trt_request(self, lora_config: Optional[trtllm.LoraConfig] = None) -> trtllm.Request:
        return to_trt_req(self._prompt_ids, self._sampling_params, self.lora_request, lora_config)

    def _check_stops(self):
        # Coppied from
        # https://github.com/vllm-project/vllm/blob/main/vllm/engine/output_processor/stop_checker.py#L72
        new_char_count = self._new_char_count
        sampling_params = self._sampling_params
        if not new_char_count:
            return None

        for stop_str in sampling_params.stop:
            stop_string_len = len(stop_str)
            # Avoid searching already-searched text.
            stop_index = self._generation_str.find(stop_str, -new_char_count - stop_string_len)
            if stop_index == -1:
                continue

            if sampling_params.include_stop_str_in_output:
                # Truncate to end of stop string.
                stop_index += stop_string_len
                if stop_index >= len(self._generation_str):
                    # No truncation required.
                    return stop_str

            # Truncate the output text to either the beginning
            # or end of the stop string.
            self._generation_str = self._generation_str[:stop_index]
            return stop_str
        return None

    def __repr__(self):
        return f"TrtRequest({[f'{k}={v}' for k, v in self.__dict__.items() if k != '_tokenizer']})"

    def __str__(self):
        return repr(self)


# TODO after INFE-1942 is fix, remove convert_prompt_ids_to_tokens and detokenize_incrementally,
#      and import them from huggingface


def convert_prompt_ids_to_tokens(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    prompt_ids: List[int],
    skip_special_tokens: bool = False,
) -> Tuple[List[str], int, int]:
    """Converts the prompt ids to tokens and returns the tokens and offsets
    for incremental detokenization.

    Note that not all tokens are converted to strings. Only the tokens that
    are necessary for incremental detokenization are converted to strings.
    """
    # We do not need to convert the whole prompt to tokens.
    # Offset a little more in case we have special tokens.
    prompt_ids_subset = prompt_ids[-INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET - 2 :]
    new_tokens = tokenizer.convert_ids_to_tokens(prompt_ids_subset, skip_special_tokens=skip_special_tokens)
    read_offset = len(new_tokens)
    prefix_offset = max(read_offset - INITIAL_INCREMENTAL_DETOKENIZATION_OFFSET, 0)
    return new_tokens, prefix_offset, read_offset


# Based on
# https://github.com/huggingface/text-generation-inference/blob/v0.9.4/server/text_generation_server/models/model.py#L62C9-L62C15
# under Apache 2.0 license
def detokenize_incrementally(
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
    all_input_ids: List[int],
    prev_tokens: Optional[List[str]],
    prefix_offset: int,
    read_offset: int,
    skip_special_tokens: bool = False,
    spaces_between_special_tokens: bool = True,
) -> Tuple[List[str], str, int, int]:
    """Detokenizes the input ids incrementally and returns the new tokens
    and the new text.

    If `prev_tokens` is None, this function will convert the input ids to
    tokens and return the tokens and the new text. Otherwise, it will return the
    new tokens and the new text.

    This function will also return the new prefix offset and the new read
    offset to be used in the next iteration.

    The offsets are necessary to defeat cleanup algorithms in the decode which
    decide to add a space or not depending on the surrounding ids.

    Args:
        tokenizer: The tokenizer to use.
        all_input_ids: The input ids. The last id is the new token id.
        prev_tokens: The previous tokens. If None, this function will convert
            the input ids to tokens and return the tokens and the new text.
        prefix_offset: The prefix offset.
        read_offset: The read offset.
        skip_special_tokens: Whether to skip special tokens.
        spaces_between_special_tokens: Whether to add spaces between special
            tokens.
    """
    new_token_id = all_input_ids[-1]
    # This is the first iteration for this sequence
    is_first_iter = prev_tokens is None
    if is_first_iter:
        (prev_tokens, prefix_offset, read_offset) = convert_prompt_ids_to_tokens(
            tokenizer, all_input_ids[:-1], skip_special_tokens=skip_special_tokens
        )
    assert prev_tokens is not None

    # If the new token id is out of bounds, return an empty string.
    if new_token_id >= len(tokenizer):
        new_tokens = [""]
    else:
        # Put new_token_id in a list so skip_special_tokens is respected
        new_tokens = tokenizer.convert_ids_to_tokens([new_token_id], skip_special_tokens=skip_special_tokens)
        if isinstance(new_tokens, str):
            new_tokens = [new_tokens]
    output_tokens = prev_tokens + new_tokens

    # If this is the first iteration, return all tokens.
    if is_first_iter:
        new_tokens = output_tokens

    # The prefix text is necessary only to defeat cleanup algorithms in
    # the decode which decide to add a space or not depending on the
    # surrounding ids.
    if tokenizer.is_fast or not tokenizer.get_added_vocab():
        prefix_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:read_offset])
        new_text = tokenizer.convert_tokens_to_string(output_tokens[prefix_offset:])
    else:
        prefix_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:read_offset],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )
        new_text = _convert_tokens_to_string_with_added_encoders(
            tokenizer,
            output_tokens[prefix_offset:],
            skip_special_tokens=skip_special_tokens,
            spaces_between_special_tokens=spaces_between_special_tokens,
        )

    if len(new_text) <= len(prefix_text) or new_text.endswith("�"):
        # utf-8 char at the end means it's a potential unfinished byte sequence
        # from byte fallback tokenization.
        # If it's in the middle, it's probably a real invalid id generated
        # by the model
        return new_tokens, "", prefix_offset, read_offset

    new_text = new_text[len(prefix_text) :]
    return new_tokens, new_text, read_offset, len(output_tokens)
