from concurrent.futures import ThreadPoolExecutor
from typing import Iterator, List, Tuple

import anthropic
import openai


prompt_template = "Tell me a short joke about {topic}"
anthropic_template = f"Human:\n\n{prompt_template}\n\nAssistant:"
client = openai.OpenAI()
async_client = openai.AsyncOpenAI()
anthropic_client = anthropic.Anthropic()

def call_chat_model(messages: List[dict]) -> str:
    response = client.chat.completions.create(
        model="gpt-3.5-turbo", 
        messages=messages,
    )
    return response.choices[0].message.content

def invoke_chain(topic: str) -> str:
    print(f"Input: {topic}")
    prompt_value = prompt_template.format(topic=topic)
    print(f"Formatted prompt: {prompt_value}")
    messages = [{"role": "user", "content": prompt_value}]
    output = call_chat_model(messages)
    print(f"Output: {output}")
    return output

def stream_chat_model(messages: List[dict]) -> Iterator[str]:
    stream = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
        stream=True,
    )
    for response in stream:
        content = response.choices[0].delta.content
        if content is not None:
            yield content

def stream_chain(topic: str) -> Iterator[str]:
    print(f"Input: {topic}")
    prompt_value = prompt.format(topic=topic)
    print(f"Formatted prompt: {prompt_value}")
    stream = stream_chat_model([{"role": "user", "content": prompt_value}])
    for chunk in stream:
        print(f"Token: {chunk}", end="")
        yield chunk

def batch_chain(topics: list) -> list:
    with ThreadPoolExecutor(max_workers=5) as executor:
        return list(executor.map(invoke_chain, topics))

def call_llm(prompt_value: str) -> str:
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_value,
    )
    return response.choices[0].text

def invoke_llm_chain(topic: str) -> str:
    print(f"Input: {topic}")
    prompt_value = promtp_template.format(topic=topic)
    print(f"Formatted prompt: {prompt_value}")
    output = call_llm(prompt_value)
    print(f"Output: {output}")
    return output

def call_anthropic(prompt_value: str) -> str:
    response = anthropic_client.completions.create(
        model="claude-2",
        prompt=prompt_value,
        max_tokens_to_sample=256,
    )
    return response.completion   

def invoke_anthropic_chain(topic: str) -> str:
    print(f"Input: {topic}")
    prompt_value = anthropic_template.format(topic=topic)
    print(f"Formatted prompt: {prompt_value}")
    output = call_anthropic(prompt_value)
    print(f"Output: {output}")
    return output

async def ainvoke_anthropic_chain(topic: str) -> str:
    ...

def stream_anthropic_chain(topic: str) -> Iterator[str]:
    ...

def batch_anthropic_chain(topics: List[str]) -> List[str]:
    ...

def invoke_configurable_chain(
    topic: str, 
    *, 
    model: str = "chat_openai"
) -> str:
    if model == "chat_openai":
        return invoke_chain(topic)
    elif model == "openai":
        return invoke_llm_chain(topic)
    elif model == "anthropic":
        return invoke_anthropic_chain(topic)
    else:
        raise ValueError(
            f"Received invalid model '{model}'."
            " Expected one of chat_openai, openai, anthropic"
        )

def stream_configurable_chain(
    topic: str, 
    *, 
    model: str = "chat_openai"
) -> Iterator[str]:
    if model == "chat_openai":
        return stream_chain(topic)
    elif model == "openai":
        # Note we haven't implemented this yet.
        return stream_llm_chain(topic)
    elif model == "anthropic":
        # Note we haven't implemented this yet
        return stream_anthropic_chain(topic)
    else:
        raise ValueError(
            f"Received invalid model '{model}'."
            " Expected one of chat_openai, openai, anthropic"
        )

def batch_configurable_chain(
    topics: List[str], 
    *, 
    model: str = "chat_openai"
) -> List[str]:
    ...

async def abatch_configurable_chain(
    topics: List[str], 
    *, 
    model: str = "chat_openai"
) -> List[str]:
    ...

def invoke_chain_with_fallback(topic: str) -> str:
    try:
        return invoke_chain(topic)
    except Exception:
        return invoke_anthropic_chain(topic)

async def ainvoke_chain_with_fallback(topic: str) -> str:
    try:
        return await ainvoke_chain(topic)
    except Exception:
        return ainvoke_anthropic_chain(topic)

async def batch_chain_with_fallback(topics: List[str]) -> str:
    try:
        return batch_chain(topics)
    except Exception:
        return batch_anthropic_chain(topics)
