import os
import argparse
import tqdm
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('--judge_api_key', type=str, required=True, help='API key for judge model')
parser.add_argument('--judge_base_url', type=str, required=True, help='Base URL for judge model')
parser.add_argument('--judge_model_name', type=str, default='claude-3-7-sonnet-latest', help='Model name for judge model')
parser.add_argument('--tokenizer_path', type=str, default='deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', help='The tokenizer path')
parser.add_argument('--split_strategies', type=str, default='separator', choices=['llm', 'keywords', 'separator'], help='The policy to split the response')
parser.add_argument('--cache_dir', type=str, default='cache/efficiency', help='Cache directory')
parser.add_argument('--output_dir', type=str, default='final_results/efficiency/', help='Output directory for evaluation results')
parser.add_argument('--workers', type=int, default=16, help='Number of workers for evaluation')
parser.add_argument("--llm_output_dir", type=str, default='LLM_Output', help="Path to the LLM output directory.")
args = parser.parse_args()

# The path to the LLM_Output folder
llm_output_dir = args.llm_output_dir

# Iterate over all JSON files in the LLM_Output folder
for root, dirs, files in os.walk(llm_output_dir):
    for file in tqdm.tqdm(files):
        if file.endswith('.json'):
            json_path = os.path.join(root, file)
            file_name_without_ext = os.path.splitext(file)[0]
            specific_cache_dir = os.path.join(args.cache_dir, file_name_without_ext)
            specific_output_dir = os.path.join(args.output_dir, file_name_without_ext)

            judge_config = dict(
                api_key=args.judge_api_key,
                base_url=args.judge_base_url,
                model_name=args.judge_model_name,
            )

            model_config = dict(
                json_path=json_path,
                tokenizer_path=args.tokenizer_path,
                split_strategies=args.split_strategies,
                judge_config=judge_config,
                cache_dir=specific_cache_dir
            )

            command = [
                'python', 'efficiency.py',
                '--judge_api_key', args.judge_api_key,
                '--judge_base_url', args.judge_base_url,
                '--judge_model_name', args.judge_model_name,
                '--tokenizer_path', args.tokenizer_path,
                '--split_strategies', args.split_strategies,
                '--cache_dir', specific_cache_dir,
                '--output_dir', specific_output_dir,
                '--workers', str(args.workers),
                '--json_path', json_path
            ]
            try:
                subprocess.run(command, check=True)
            except subprocess.CalledProcessError as e:
                print(f'error handling {file} : {e}')
            print(f'{file} is processed')