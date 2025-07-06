import tiktoken
import json_repair
import time
import numpy as np
import os
import random
import json
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from json.decoder import JSONDecodeError
from openai import AzureOpenAI
from datetime import datetime
from pathlib import Path


PROMPT_TEMPLATE = """You are a helpful assistant, specializing in English Grammar checking.
You must check if the 'EXAMPLE' extracted from 'TEXT' is an example of Passive Language checking the context.

Here, some rules that represents a passive example:
- If the example starts with: 'Work', 'Function', 'Handle', 'Participate', 'Worked', 'Functioned', 'Handled', 'Participated'.
- If the example starts with: 'Responsible', 'Accountable'.
- If the example doesn't start with a active verb.
- If the example starts with a Noun.
- If the example starts with an abstract Noun, for example, 'supervision'.
- If the example is on passive voice.

Return 'True' if yes, or 'False' if it's not passive language.
Remember, if the example starts a active verb, return 'False'.
Don't explain or give additional information. Just return 'True' or 'False'.

'TEXT': 
'''
{text}
'''

'EXAMPLE': {ex}
"""

rate_limit = {
    "minute": int(np.floor(time.time())) // 60,
    "total_tokens": 0
}

def num_tokens_from_string(string: str, encoding_name: str="cl100k_base") -> int:
    """
    Returns the number of tokens in a text string.
    
    Args:
        string (str): The input text string.
        encoding_name (str): The name of the encoding to use. Defaults to "cl100k_base".
    
    Returns:
        int: The number of tokens in the text string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

azure_openai_client = AzureOpenAI(
    api_key=os.environ["AZURE_OPENAI_API_KEY"],
    api_version="2024-04-01-preview",
    azure_endpoint="https://leoaieng-azure-openai-instance.openai.azure.com"
)

def check_ratelimit(prompt):
    global rate_limit
    cur_minute = rate_limit["minute"]
    new_minute = int(np.floor(time.time())) // 60
    n_tokens_next = num_tokens_from_string(prompt)
    if new_minute > cur_minute:
        last_minute_tot_tokens = rate_limit["total_tokens"]
        last_minute = cur_minute
        rate_limit["minute"] = new_minute
        rate_limit["total_tokens"] = 0
        rate_limit["total_tokens"]+=n_tokens_next
        new_tot_tokens = rate_limit["total_tokens"]

        print(f"New Minute is: {new_minute}")
        print(f"New Total Tokens is: {new_tot_tokens}")
        print(f"Last Minute: {last_minute}")
        print(f"Last Total tokens: {last_minute_tot_tokens}")
    elif new_minute == cur_minute:
        if (rate_limit["total_tokens"] + n_tokens_next + 8192) >= 450000:
            tot_tokens = rate_limit["total_tokens"]
            print(f"Rate Limit Reached! Waiting for 60s..")
            print(f"Minute: {new_minute}")
            print(f"Total Tokens: {tot_tokens}")
            print(f"Next Tokens: {(n_tokens_next + 8192)}")
            time.sleep(61)
            rate_limit["minute"] = int(np.floor(time.time())) // 60
            rate_limit["total_tokens"] = 0
            rate_limit["total_tokens"]+=n_tokens_next
            new_minute = rate_limit["minute"]
            new_tot_tokens = rate_limit["total_tokens"]
            print(f"New Minute is: {new_minute}")
            print(f"New Total Tokens is: {new_tot_tokens}")
        else:
            rate_limit["total_tokens"]+=n_tokens_next

def invoke_azure_openai_with_text(prompt: dict, temperature=0.1, max_tokens=8192, max_retries=3):
    global rate_limit
    _prompt = prompt["prompt"]

    attempt=0
    while (attempt:=attempt+1) <= max_retries:
        messages = [
            {
                "role": "user",
                "content": _prompt
            }
        ]

        # Check Rate Limit
        print("Checking limit rate..")
        check_ratelimit(_prompt)
        print("Generating...")
        completion = azure_openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        # print(completion)
        
        output_text, n_output_tokens = process_response_azure_openai(completion)
        rate_limit["total_tokens"] += n_output_tokens
        if output_text:
            prompt["res"] = output_text

            return prompt
            return output_text
            
    


def execute_azure_openai_with_text_in_thread_pool(prompts, temperature=0.1, max_tokens=8192, max_workers=2):
    """
    Executes the Claude model with multiple user prompts concurrently using a thread pool.

    Args:
        prompts (list): A list with records of prompts
        temperature (float, optional): The temperature parameter for the model. Defaults to 0.1.
        max_tokens (int, optional): The maximum number of tokens to generate. Defaults to 8192.

    Returns:
        list: A list of tuples containing the JSON response, the generated text, and the CSV name.
    """
    try:
        with tqdm(total=len(prompts)) as pbar:
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(invoke_azure_openai_with_text, prompt, temperature, max_tokens) for prompt in prompts]
                # results = [future.result() for future in futures]
                for future in as_completed(futures):
                    pbar.update(1)
                    results.append(future.result())
        return results
    except Exception as e:
        print(f"Error executing thread pool. Error: {e}")
        return []


def process_response_azure_openai(response):
    res_json = json.loads(response.json())
    output_text = res_json["choices"][0]["message"]["content"]
    n_output_tokens = res_json["usage"]["completion_tokens"]

    # try:
    #     output_recs = format_recs_from_text(output_text, n_out_ex)
    #     return output_recs, n_output_tokens
    # except Exception as e:
    #     print(e)
    #     return False, n_output_tokens

    return output_text, n_output_tokens

create_prompt = lambda text, ex: PROMPT_TEMPLATE.format(text=text, ex=ex)
    
def create_prompts():
    return False
    # return prompts
        

def split_list(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))