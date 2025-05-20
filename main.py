import json
import os
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Union

from tqdm import tqdm

from utils.dataset import get_dataset_by_path
from utils.file_utils import query_gpt

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='precision',choices=['recall','precision', 'reflection_quality'], help="Path to the dataset class.")
parser.add_argument("--num_threads", type=int, default=16, help="Number of threads.")
parser.add_argument("--prompt_path", default='prompt/prompt_precision.txt', help="Path to the prompt file.")
parser.add_argument("--data_path", default=r'LLM_Output/glm-z1-air.json', help="Path to the query input file.")
parser.add_argument("--model", type=str, default='claude-3-7-sonnet-latest')
parser.add_argument("--openai_api_key", type=str, required=True, help="API key.")
parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM API.")
parser.add_argument("--cache_dir", type=str, default='cache/precision/glm-z1-air')

args = parser.parse_args()

def task(inputs: Dict[str, Union[str, Dict[str, Union[str, int]]]]) -> Dict[str, Union[Dict[str, int], List[str]]]:
    try:
        gpt_output, index = query_gpt(inputs, args)

        result = {
            "valid_outputs": gpt_output.choices[0].message.content,
            "index": index,
        }

        result.update(inputs)
        del result['query_input']
    except Exception as e:
        result = {"error_message": str(e)}
        print(result)
        return {}
    cache = True
    if cache:
        json.dump(result, open(f'./{args.cache_dir}/{result["index"]}.json', 'w'), indent=4)
    return result


if __name__ == "__main__":

    dataset_args = {
        'prompt_path': getattr(args, "prompt_path", None),
        'data_path': getattr(args, "data_path", None),
        'cache_dir': getattr(args, "cache_dir", None),
    }

    os.makedirs(args.cache_dir, exist_ok=True)

    dataset = get_dataset_by_path(args.name, dataset_args) # a list

    results = []
    query_inputs = []
    start_time = time.time()

    if args.num_threads == 0:
        progress_bar = tqdm(total=len(dataset), unit="task")
        for n, d in enumerate(dataset):
            query_inputs.append(d["query_input"])
            results.append(task(d))
            progress_bar.update(1)
        progress_bar.close()
    else:

        def update_progress(_):
            progress_bar.update(1)

        # Submit the tasks to the thread pool
        progress_bar = tqdm(total=len(dataset), unit="task")
        batch_size = args.num_threads
        for i in range(0, len(dataset), batch_size):
            # Create a thread pool with the specified number of threads
            with ThreadPoolExecutor(max_workers=args.num_threads) as executor:
                current_batch = dataset[i: i + batch_size]
                futures = [executor.submit(task, d) for d in current_batch]
                # Retrieve the results as they become available
                for future, num in zip(futures, dataset):
                    results.append(future.result())
                    progress_bar.update(1)
        progress_bar.close()

    duration = time.time() - start_time


    print(f"Total time: {duration:.2f}s")
