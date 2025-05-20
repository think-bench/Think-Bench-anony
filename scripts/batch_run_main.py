import os
import argparse
import tqdm
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='recall', choices=['recall', 'precision', 'reflection_quality'], help="Path to the dataset class.")
parser.add_argument("--prompt_path", default='prompt/prompt_recall.txt', help="Path to the prompt file.")
parser.add_argument("--num_threads", type=int, default=16, help="Number of threads.")
parser.add_argument("--model", type=str, default='claude-3-7-sonnet-latest')
parser.add_argument("--openai_api_key", type=str, required=True, help="API key.")
parser.add_argument("--llm_url", type=str, required=True, help="URL of the LLM API.")
parser.add_argument("--llm_output_dir", type=str, default='LLM_Output', help="Path to the LLM output directory.")
args = parser.parse_args()

# The path to the LLM_Output folder
llm_output_dir = args.llm_output_dir

# Iterate over all JSON files in the LLM_Output folder
for root, dirs, files in os.walk(llm_output_dir):
    for file in tqdm.tqdm(files):
        if file.endswith('.json'):
            data_path = os.path.join(root, file)
            file_name_without_ext = os.path.splitext(file)[0]
            cache_dir = os.path.join('cache_Prompt', args.name, file_name_without_ext)

            command = [
                'python', 'main.py',
                '--name', args.name,
                '--num_threads', str(args.num_threads),
                '--prompt_path', args.prompt_path,
                '--data_path', data_path,
                '--model', args.model,
                '--openai_api_key', args.openai_api_key,
                '--llm_url', args.llm_url,
                '--cache_dir', cache_dir
            ]

            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error while executing {command}: {e}")
            print(f"{file} is processed")