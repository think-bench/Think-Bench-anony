# THINK-Bench: Evaluating Thinking Efficiency and Chain-of-Thought Quality of Large Reasoning Models

Official repository for "THINK-Bench: Evaluating Thinking Efficiency and Chain-of-Thought Quality of Large Reasoning Models".

For more details, please refer to the project page with dataset exploration and visualization tools.

[[🤗Huggingface Dataset](https://huggingface.co/datasets/zhiyuan218/Think-Bench)] [[ModelScope Dataset](https://www.modelscope.cn/datasets/zhiyuan218/Think-Bench)] [[Visualization](https://huggingface.co/datasets/zhiyuan218/Think-Bench/viewer)]

## 👀 About Think-Bench

Reasoning models have achieved significant advancements in handling complex tasks, often surpassing traditional large language models.  However, the challenge of overthinking remains common, significantly hindering computational efficiency.  This issue arises as models produce an excess of redundant tokens that contribute little to the accuracy of answers, particularly in simpler tasks, resulting in considerable waste of resources.

<p align="center">
    <img src="image/pipeline.png" width="90%"> <br>
</p>

To address this issue systematically, we introduce Think-Bench, a benchmark designed to evaluate the thinking efficiency of large reasoning models (LRMs).  We propose a new efficiency metric and conduct a comprehensive analysis of LRMs from multiple aspects, including the reasoning process and chain-of-thought (CoT) characteristics.  

<p align="center">
    <img src="image/dataset overview.png" width="90%"> <br>
</p>

Leveraging the Think-Bench benchmark and a novel evaluation strategy, we conduct a comprehensive analysis of large reasoning models (LRMs), uncovering several key insights: (1) Most LRMs tend to **overthink on simple tasks**, generating unnecessarily long reasoning chains, while they show higher efficiency in hard problems; (2) **There is a significant trade-off between efficiency and CoT quality among different models.**    Grok-3-mini-beta achieves the highest efficiency score, while models like Qwen3-235b-a22b and Ernie-x1-turbo-32k stand out in CoT quality; (3) **Models show task heterogeneity in different disciplinary tasks.** Mathematical tasks generally have high token consumption and low reasoning efficiency, while chemistry and physics tasks show higher reasoning efficiency and lower token occupancy rate. We hope Think-Bench serves as an important benchmark for optimizing the performance of large reasoning models in the future.

<p align="center">
    <img src="image/radar.png" width="60%"> <br>
</p>


## 📚 Dataset

We release the Think-Bench dataset on [huggingface](https://huggingface.co/datasets/zhiyuan218/Think-Bench) and [modelscope](https://www.modelscope.cn/datasets/zhiyuan218/Think-Bench).
You can download the dataset from the [Huggingface](https://huggingface.co/datasets/zhiyuan218/Think-Bench) or [ModelScope](https://www.modelscope.cn/datasets/zhiyuan218/Think-Bench).

### Data Usage

You can download the dataset from the [🤗 Huggingface](https://huggingface.co/datasets/zhiyuan218/Think-Bench) by the following command (make sure that you have installed [related packages](https://huggingface.co/docs/datasets/quickstart)):

```python
from datasets import load_dataset

dataset = load_dataset("zhiyuan218/Think-Bench")
```

Or You can download the dataset from the [ModelScope](https://www.modelscope.cn/datasets/zhiyuan218/Think-Bench) by the following command (make sure that you have installed [related packages](https://www.modelscope.cn/docs/intro/quickstart)):

```python
from modelscope.msdatasets import MsDataset
dataset =  MsDataset.load('zhiyuan218/Think-Bench')
```

## Inference

To run the inference with the model, you can use the following command:
```bash
python eval_LRM.py --input_file YOUR_DATASET_PATH --output_file LLM_Output/YOUR_MODEL_NAME.json --openai_api_key YOUR_API_KEY --llm_url YOUR_LLM_URL --model YOUR_MODEL_NAME 
```
If you use prompt engineering, you can use the following command:
```bash
python eval_LRM.py --input_file YOUR_DATASET_PATH --output_file LLM_Output_Prompt/YOUR_MODEL_NAME.json --openai_api_key YOUR_API_KEY --llm_url YOUR_LLM_URL --model YOUR_MODEL_NAME --prompt
```

After LRM inference, you are expected to obtain a `cache/` directory like this:
```
    📂 cache
    ┣━━ 📂 YOUR_MODEL_NAME
    ┃    ┣━━ 📄 1.json
    ┃    ┣━━ 📄 2.json
    ┃    ┗━━ 📄...
    ┣━━ 📂 YOUR_MODEL_NAME
    ┃   
    ┗━━ 📂 YOUR_MODEL_NAME
```

## Evaluation

To calculate the nine metrics (Efficiency, Reflection Tokens, Useful Tokens, Tokens, Thought Num,  Reflection Quality, Precision, Recall, Accuracy), please follow the following steps:
1. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```
2. Run the evaluation script.

     You can either run the single metrics(recall, precision, reflection_quality) for one the models. For example, to evaluate recall:
     ```bash
     python main.py --name recall --prompt_path prompt/prompt_recall.txt --data_path YOUR_MODEL_INFER_DATA_PATH --openai_api_key YOUR_API_KEY --llm_url YOUR_LLM_URL --cache_dir cache/recall/YOUR_MODEL_NAME
     ```

     You can run the efficiency script for one the models. For example, to evaluate efficiency:
     ```bash
     python efficiency.py --judge_api_key YOUR_API_KEY --judge_base_url YOUR_JUDGE_URL --json_path YOUR_MODEL_INFER_DATA_PATH --cache_dir cache/efficiency/YOUR_MODEL_NAME --output_dir final_results/efficiency/YOUR_MODEL_NAME/
     ```

     Or you can run the single metrics(recall, precision, reflection_quality) for all the models in one directory. For example, to evaluate recall:

     ```bash
     python scripts/batch_run_main.py --name recall --prompt_path prompt/prompt_recall.txt --llm_output_dir YOUR_MODEL_INFER_DATA_PATH --openai_api_key YOUR_API_KEY --llm_url YOUR_LLM_URL
     ```

     Or you can run the efficiency script for all the models in one directory. For example, to evaluate efficiency:
     ```bash
     python scripts/batch_run_efficiency.py --judge_api_key YOUR_API_KEY --judge_base_url YOUR_JUDGE_URL --llm_output_dir YOUR_MODEL_INFER_DATA_PATH
     ```

     After Claude 3.7 Sonnet evaluation, you are expected to obtain a `cache/` directory like this:
    ```
      📂 cache
       ┣━━ 📂 efficiency
       ┃    ┗━━ 📂 YOUR_MODEL_NAME
       ┃         ┣━━ 📄 1.json
       ┃         ┣━━ 📄 2.json
       ┃         ┗━━ 📄 ...
       ┣━━ 📂 reflection_quality
       ┃    ┗━━ 📂 YOUR_MODEL_NAME
       ┣━━ 📂 recall
       ┃    ┗━━ 📂 YOUR_MODEL_NAME
       ┗━━ 📂 precision
            ┗━━ 📂 YOUR_MODEL_NAME
    ```

3. Calculate the metrics.

     We cache the evaluation results of all the questions in the cache dir. Here we read the results from the cache dir and calculate the metrics. 

     For example, to calculate recall:
     ```bash
     python final_score/recall.py --cache_dir cache/recall --save_path final_results
     ```
     


## Acknowledgements
Our project referred to the following repositories:
- [MME-Cot](https://github.com/MME-Benchmarks/MME-CoT)
- [evalscope](https://github.com/modelscope/evalscope)
