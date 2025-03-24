import boto3
import json
from typing import Any, Dict, Generator, Optional, Tuple, AsyncGenerator
from pydantic import Field
from llama_index.core.llms import LLM, ChatMessage, ChatResponse, CompletionResponse, LLMMetadata

class BedrockLLM(LLM):
    # model_id: str = Field(default="meta.llama3-3-70b-instruct-v1:0")
    model_id: str = Field(default="deepseek.r1-v1:0")
    region: str = Field(default="us-east-1")
    client: Any = Field(default=None, exclude=True)

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.client = boto3.client("bedrock-runtime", region_name=self.region)

    @classmethod
    def class_name(cls) -> str:
        return "BedrockLLM"

    def _format_prompt(self, prompt: str) -> str:
        return f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""

    def _invoke_bedrock(self, body: Dict[str, Any]) -> Dict[str, Any]:
        response = self.client.invoke_model(
            modelId=self.model_id,
            contentType="application/json",
            accept="application/json",
            body=json.dumps(body)
        )
        return json.loads(response["body"].read())

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        formatted_prompt = self._format_prompt(prompt)
        body = {
            "prompt": formatted_prompt,
            "max_gen_len": kwargs.get("max_tokens", 512),
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.9)
        }
        result = self._invoke_bedrock(body)
        return CompletionResponse(text=result["generation"])

    # Required abstract method implementations
    def stream_complete(self, prompt: str, **kwargs: Any) -> Generator[CompletionResponse, None, None]:
        response = self.complete(prompt, **kwargs)
        yield response

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        return self.complete(prompt, **kwargs)

    async def astream_complete(self, prompt: str, **kwargs: Any) -> AsyncGenerator[CompletionResponse, None]:
        response = self.complete(prompt, **kwargs)
        yield response

    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = "\n".join([f"{msg.role}: {msg.content}" for msg in messages])
        completion = self.complete(prompt, **kwargs)
        return ChatResponse(message=ChatMessage(role="assistant", content=completion.text))

    async def achat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        return self.chat(messages, **kwargs)

    def stream_chat(self, messages: list[ChatMessage], **kwargs: Any) -> Generator[ChatResponse, None, None]:
        response = self.chat(messages, **kwargs)
        yield response

    async def astream_chat(self, messages: list[ChatMessage], **kwargs: Any) -> AsyncGenerator[ChatResponse, None]:
        response = self.chat(messages, **kwargs)
        yield response

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=8192,
            num_output=512,
            is_chat_model=True,
            is_function_calling_model=False,
            model_name=self.model_id
        )