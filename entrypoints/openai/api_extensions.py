import re
from enum import Enum
from http import HTTPStatus
from typing import Annotated, ClassVar, Dict, List, Literal, Optional, Union

from pydantic import AfterValidator, BaseModel, Field, StrictFloat, StrictInt, StrictStr, field_validator
from vllm.entrypoints.openai.protocol import ChatCompletionRequest, CompletionRequest, ErrorResponse, ResponseFormat
#from vllm.entrypoints.openai.serving_chat import OpenAIServingChat as OpenAIServingChatOriginal
#from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion as OpenAIServingCompletionOriginal
from vllm.sampling_params import SamplingParams

# Filter out tag chars to mitigate direct and indirect LLM prompt injection attacks
UNSAFE_TAG_CHARACTERS_PATTERN = re.compile(r'[\U000e0020-\U000e007f]')


def filter_tag_characters(prompt: str):
    return UNSAFE_TAG_CHARACTERS_PATTERN.sub('', prompt)


PromptValue = Annotated[StrictStr, AfterValidator(filter_tag_characters)]


class Role(str, Enum):
    assistant = "assistant"
    user = "user"
    system = "system"
    tool = "tool"
    function = "function"


class ResponseFormat(BaseModel):
    type: Literal["text", "json_object"] = Field(
        "text",
        description="Must be one of `text` or `json_object`.",
        example="json_object",
    )


# list for vllm parameters needed by GenAI
# implicit parameter support: functional but hidden from MIM openapi doc
nim_genai_allowlist = ["ignore_eos", "min_tokens"]

# list for OpenAI parameters not supported by MIM
nim_api_openai_denylist = [
    "best_of",
    "response_format",
    "echo",
]

# list for vllm api extensions removed from MIM openapi doc
nim_api_vllm_ext_denylist = [
    # vllm parameters removed from openapi, unsupprted by MIM
    'use_beam_search',
    'early_stopping',
    'detokenize',
    'logits_processors',
    'truncate_prompt_tokens',
    'add_generation_prompt',
    'guided_choice',
    'image',
    'guided_json',
    'guided_regex',
    'guided_grammar',
    'guided_decoding_backend',
    # vllm parameters removed from openapi, supprted by MIM
    'length_penalty',
    'stop_token_ids',
    'min_tokens',
    'skip_special_tokens',
    'spaces_between_special_tokens',
    'include_stop_str_in_output',
]

# list for vllm extensions moved to MIM nvext
nim_api_nvext_list = [
    'ignore_eos',
    'repetition_penalty',
    'top_k',
]


class NIMLLMChatCompletionMessage(BaseModel):
    role: Role = Field(description="The role of the message's author.")
    content: PromptValue = Field(description="The contents of the message.")


class NVExt(BaseModel):
    ignore_eos: Optional[bool] = Field(
        None,
        description="Whether to ignore End of Sequence (EOS) tokens.",
    )
    repetition_penalty: Optional[StrictFloat] = Field(
        1.0,
        description="How much to penalize tokens based on how frequently they occur in the text. A value of 1 means no penalty, while values larger than 1 discourage and values smaller encourage.",
        gt=0,
        le=2,
    )
    top_k: StrictInt = Field(
        -1,
        description="How many of the top tokens to sample from. Must be -1 or greater than or equal to 1, and cannot be null. If not set, then the default is -1 which disables top_k sampling (greedy).",
        ge=-1,
    )

    @field_validator('top_k')
    def check_top_k(cls, value):
        if value != -1 and value < 1:
            raise ValueError('top_k must be -1 or greater than or equal to 1')
        return value


class CombinedRequestParameters(BaseModel):

    # Redefine the 'model' field with a new description
    model: str = Field(
        ...,
        description="The model to use.",
    )
    ignore_eos: Optional[bool] = Field(default=False, exclude=True)
    repetition_penalty: Optional[float] = Field(default=1.0)
    top_k: Optional[int] = Field(default=-1)
    add_generation_prompt: Optional[bool] = Field(
        default=True,
        description="If true, the generation prompt will be added to the chat template. This is a parameter used by chat template in tokenizer config of the model.",
    )
    guided_choice: Optional[List[str]] = Field(
        default=None, description="If specified, the output will be exactly one of the choices."
    )
    image: Optional[Union[str, None]] = Field(default=None, description="The image as a string or null.")
    length_penalty: Optional[Union[float, None]] = Field(
        default=1.0, description="Length Penalty, can be a number or null."
    )
    stop_token_ids: Optional[Union[List[int], None]] = Field(
        default=None, description="Stop Token Ids, can be an array of integers or null."
    )
    min_tokens: Optional[Union[int, None]] = Field(default=0, description="Min Tokens, can be an integer or null.")
    skip_special_tokens: Optional[Union[bool, None]] = Field(
        default=True, description="Skip Special Tokens, can be a boolean or null."
    )
    spaces_between_special_tokens: Optional[Union[bool, None]] = Field(
        default=True, description="Spaces Between Special Tokens, can be a boolean or null."
    )
    include_stop_str_in_output: Optional[Union[bool, None]] = Field(
        default=False,
        description="Whether to include the stop string in the output. This is only applied when the stop or stop_token_ids is set. Can be a boolean or null.",
    )
    guided_json: Optional[Union[str, dict, BaseModel, None]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema. Can be a string, an object, a reference to a BaseModel, or null.",
    )
    guided_regex: Optional[Union[str, None]] = Field(
        default=None, description="If specified, the output will follow the regex pattern. Can be a string or null."
    )
    guided_grammar: Optional[Union[str, None]] = Field(
        default=None,
        description="If specified, the output will follow the context-free grammar. Can be a string or null.",
    )
    guided_decoding_backend: Optional[Union[str, None]] = Field(
        default=None,
        description="If specified, will override the default guided decoding backend of the server for this specific request. If set, must be either 'outlines' / 'lm-format-enforcer'. Can be a string or null.",
    )

    class Config:
        # Use this to exclude fields from OpenAPI schema
        @staticmethod
        def json_schema_extra(schema, model):
            properties = schema.get('properties', {})
            for field in nim_api_nvext_list + nim_api_vllm_ext_denylist:
                if field in properties:
                    del properties[field]

    best_of: Optional[Union[int, None]] = Field(
        None,
        title="Best Of",
        description=(
            "This is currently unsupported. Generates best_of completions "
            "server-side and returns the 'best' (the one with the highest log probability "
            "per token). Results cannot be streamed. When used with n, best_of controls "
            "the number of candidate completions and n specifies how many to return - "
            "best_of must be greater than n."
        ),
    )
    response_format: ResponseFormat = Field(
        ...,
        description=(
            "This is currently unsupported. "
            "An object specifying the format that the model must output."
            "Setting to `{ \"type\": \"json_object\" }` enables JSON mode, which guarantees the message the model generates is valid JSON.\n\n"
        ),
    )
    echo: Optional[bool] = Field(
        default=False,
        description=(
            "This is currently unsupported. "
            "If true, the new message will be prepended with the last message if they belong to the same role."
        ),
    )
    frequency_penalty: Optional[StrictFloat] = Field(
        default=0.0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on their existing frequency in the text so far, decreasing the model's likelihood to repeat the same line verbatim.",
        ge=-2.0,
        le=2.0,
    )
    logit_bias: Optional[Dict[str, float]] = Field(
        default=None,
        description="Modify the likelihood of specified tokens appearing in the completion. Accepts a JSON object that maps tokens (specified by their token ID in the tokenizer) to an associated bias value from -100 to 100. Mathematically, the bias is added to the logits generated by the model prior to sampling. The exact effect will vary per model, but values between -1 and 1 should decrease or increase likelihood of selection; values like -100 or 100 should result in a ban or exclusive selection of the relevant token.",
    )
    n: Optional[StrictInt] = Field(
        default=1, description="How many completions to generate for each prompt.", ge=1, le=128
    )
    presence_penalty: Optional[StrictFloat] = Field(
        default=0.0,
        description="Number between -2.0 and 2.0. Positive values penalize new tokens based on whether they appear in the text so far, increasing the model's likelihood to talk about new topics.",
        ge=-2.0,
        le=2.0,
    )
    seed: Optional[StrictInt] = Field(
        default=None,
        description="Changing the seed will produce a different response with similar characteristics. Fixing the seed will reproduce the same results if all other parameters are also kept constant.",
        ge=-9223372036854775808,
        le=9223372036854775807,
    )
    stop: Optional[Union[str, List[str]]] = Field(
        default_factory=list, description="Sequences where the API will stop generating further tokens."
    )
    stream: Optional[bool] = Field(
        default=False,
        description="If set, partial message deltas will be sent, like in ChatGPT. Tokens will be sent as data-only [server-sent events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events#Event_stream_format) as they become available, with the stream terminated by a `data: [DONE]`",
    )
    temperature: Optional[StrictFloat] = Field(
        default=1.0,
        description="What sampling temperature to use, between 0 and 2. Higher values like 0.8 will make the output more random, while lower values like 0.2 will make it more focused and deterministic.",
        ge=0.0,
        le=2.0,
    )
    top_p: Optional[StrictFloat] = Field(
        default=1.0,
        description="An alternative to sampling with temperature, called nucleus sampling, where the model considers the results of the tokens with top_p probability mass. So 0.1 means only the tokens comprising the top 10% probability mass are considered. We generally recommend altering this or `temperature` but not both.",
        gt=0.0,
        le=1.0,
    )

    nvext: Optional[NVExt] = Field(
        None,
        description="Extension dictionary for MIM API.",
    )

    # Overwrites and hides `best_of` and `response_format`
    # TODO: Add them back once they are supported (https://jirasw.nvidia.com/browse/INFE-2136) (https://jirasw.nvidia.com/browse/INFE-2135)
    best_of: ClassVar[Optional[int]] = None
    response_format: ClassVar[ResponseFormat] = None

    # Override this method to pull the properties that are nested under nvext
    # instead of in the base request
    def to_sampling_params(self) -> SamplingParams:
        sampling_params = super().to_sampling_params()
        if self.nvext is not None:
            sampling_params.ignore_eos = self.nvext.ignore_eos
            sampling_params.repetition_penalty = self.nvext.repetition_penalty
            sampling_params.top_k = self.nvext.top_k
        return sampling_params

    @field_validator('frequency_penalty', 'n', 'presence_penalty', 'seed', 'temperature', 'top_p')
    def set_default_if_none_float(cls, v, field):
        if v is None:
            return cls.model_fields[field.field_name].default
        return v

    def __init__(self, **data):
        nim_api_vllm_ext_denylist_filtered = [
            item for item in nim_api_vllm_ext_denylist if item not in nim_genai_allowlist
        ]
        nim_api_nvext_list_filtered = [item for item in nim_api_nvext_list if item not in nim_genai_allowlist]
        for unsupported_param in (
            nim_api_openai_denylist + nim_api_vllm_ext_denylist_filtered + nim_api_nvext_list_filtered
        ):
            if unsupported_param in data:
                raise ValueError(f"The field {unsupported_param} is unsupported")
        super().__init__(**data)


class NIMLLMChatCompletionRequest(CombinedRequestParameters, ChatCompletionRequest):
    messages: List[NIMLLMChatCompletionMessage] = Field(
        ..., description="A list of messages comprising the conversation so far.", min_length=1
    )
    max_tokens: Optional[StrictInt] = Field(
        default=None, description="The maximum number of tokens that can be generated.", ge=1
    )
    logprobs: Optional[bool] = Field(
        default=False,
        description="Whether to return log probabilities of the output tokens or not. If true, returns the log probabilities of each output token returned in the `content` of `message`.",
    )
    top_logprobs: Optional[StrictInt] = Field(
        default=None,
        description="An integer specifying the number of most likely tokens to return at each token position, each with an associated log probability. `logprobs` must be set to `true` if this parameter is used.",
        ge=1,
        le=20,
    )


class NIMLLMCompletionRequest(CombinedRequestParameters, CompletionRequest):
    prompt: Union[List[int], List[List[int]], PromptValue, List[PromptValue]]
    max_tokens: Optional[StrictInt] = Field(
        default=16, description="The maximum number of tokens that can be generated.", ge=1
    )
    logprobs: Optional[int] = Field(
        default=None,
        ge=0,
        le=5,
        description="Include the log probabilities on the `logprobs` most likely output tokens, as well the chosen tokens. For example, if `logprobs` is 5, the API will return a list of the 5 most likely tokens. The API will always return the `logprob` of the sampled token, so there may be up to `logprobs+1` elements in the response.",
    )
    suffix: Optional[str] = Field(
        default=None, description="The suffix that comes after a completion of inserted text."
    )

    @field_validator('max_tokens')
    def set_default_if_none_float_completion(cls, v, field):
        if v is None:
            return cls.model_fields[field.field_name].default
        return v


class NIMLLMVersionResponse(BaseModel):
    release: str = Field(description="The product release version of MIM for LLMs.")
    api: str = Field(description="The semver API version running inside the MIM.")


class NIMHealthSuccessResponse(BaseModel):
    object: str = "health.response"
    message: str


# A Mixin to override some of the methods defined in the base `OpenAIServing` class in vllm
class NIMOpenAIServingMixin:
    async def _check_model(self, request) -> Optional[ErrorResponse]:
        if request.model in self.served_model_names:
            return None
        # vllm expects just the Lora adapter name in the `model` field in the openAI request
        # MIM expects the base model prepended like `{base_model}:{lora_adapter}`
        # Therefore passing just the lora adapter name should result in a model not found 404 error
        # This method explicitly overrides the `_check_model` function defined in vllm
        # and the following lines are commented out to explicitly disallow just the lora adapter name
        # If this is not done then the request gets routed to just the base model which is incorrect behavior

        if request.model in [lora.lora_name for lora in self.get_lora_requests()]:
            return None
        return self.create_error_response(
            message=f"The model `{request.model}` does not exist.",
            err_type="NotFoundError",
            status_code=HTTPStatus.NOT_FOUND,
        )


# class OpenAIServingChat(NIMOpenAIServingMixin, OpenAIServingChatOriginal):
#     pass


# class OpenAIServingCompletion(NIMOpenAIServingMixin, OpenAIServingCompletionOriginal):
#     pass
