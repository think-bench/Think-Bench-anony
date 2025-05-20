import re
import time
import openai
from openai import OpenAI
import re
openai.request_timeout = 3600

def extract_think_content(content):
    think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)
    match = think_pattern.search(content)
    if match:
        reasoning_content = match.group(1)
        content = think_pattern.sub('', content).strip()
        return content, reasoning_content
    else:
        return content, ""


def deepseek(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
            )
            if response.choices[0].message.content is not None:
                success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None
    return response.choices[0].message.content, response.choices[0].message.reasoning_content

def qwq(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            completion  = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None

    reasoning_content = ""  # define reasoning content
    answer_content = ""     # define answer content
    is_answering = False   # judge if the model is answering
    try:
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    # answer content
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content
    except Exception as e:
        print(inputs["index"])
        print(f'ERROR: {e}')
        return None, None
    return answer_content, reasoning_content

def qwen3(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            completion  = client.chat.completions.create(
                model=args.model,
                messages=messages,
                extra_body={"enable_thinking": True},
                stream=True,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None

    reasoning_content = ""  # define reasoning content
    answer_content = ""     # define answer content
    is_answering = False   # judge if the model is answering
    try:
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    # answer content
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content
    except Exception as e:
        print(inputs["index"])
        print(f'ERROR: {e}')
        return None, None
    return answer_content, reasoning_content


def claude(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
            )
            content, reasoning_content = extract_think_content(response.choices[0].message.content)
            if content is not None and reasoning_content is not None:
                success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None
    
    return content, reasoning_content


def grok3(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None
    
    return response.choices[0].message.content, response.choices[0].message.reasoning_content

def ernie(inputs, args): 
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except Exception as e:
            print(f'ERROR: {e}')
            return None, None
    reasoning_content = ""
    content = ""
    try:
        for chunk in response:
            if hasattr(chunk.choices[0].delta, 'reasoning_content') and chunk.choices[0].delta.reasoning_content:
                reasoning_content += chunk.choices[0].delta.reasoning_content
            else:
                if chunk.choices[0].delta.content is not None:
                    content += chunk.choices[0].delta.content
    except Exception as e:
        print(inputs["index"])
        print(f'ERROR: {e}')
        return None, None
    return content, reasoning_content

def glm(inputs, args):
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                messages=messages,
                stream=True,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None
    content = ""
    try:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                content += chunk.choices[0].delta.content
        contents, reasoning_content = extract_think_content(content)
    except Exception as e:
        print(inputs["index"])
        print(f'ERROR: {e}')
        return None, None
        
    return contents, reasoning_content

def deepseek_distill(inputs, args): 
    client = OpenAI(api_key=args.openai_api_key, base_url=args.llm_url)
    system_messages = "You are an AI assistant that helps people solve their questions."
    messages = [
        {
            "role": "system",
            "content": system_messages
        },
        {
            "role": "user",
            "content": inputs["query_input"]
        }
    ]
    success = True
    while success:
        try:
            response = client.chat.completions.create(
                model=args.model,
                stream=True,
                messages=messages,
            )
            success = False
        except openai.RateLimitError as e:
            print('Rate limit exceeded, waiting for 60 seconds...')
            print(f'ERROR: {e}')
            time.sleep(60)
        except openai.APIConnectionError as e:
            print('API connection error, waiting for 10 seconds...')
            print(f'ERROR: {e}')
            time.sleep(10)
        except Exception as e:
            if 'RequestTimeOut' in str(e):
                print(f'ERROR: {e}')
                time.sleep(5)
            else:
                print(f'ERROR: {e}')
                return None, None
    reasoning_content = ""  # define reasoning content
    answer_content = ""     # define answer content
    is_answering = False   # judge if the model is answering
    try:
        for chunk in response:
            if chunk.choices:
                delta = chunk.choices[0].delta
                # reasoning content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content != None:
                    reasoning_content += delta.reasoning_content
                else:
                    # answer content
                    if delta.content != "" and is_answering is False:
                        is_answering = True
                    answer_content += delta.content
    except Exception as e:
        print(inputs["index"])
        print(f'ERROR: {e}')
        return None, None
    return answer_content, reasoning_content
