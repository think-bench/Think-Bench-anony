import json
import os
import argparse
from collections import defaultdict
import tqdm
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.file_utils import extract_json_string

def calculate_reflection_quality(data_list):
    """
    Calculate reflection quality metrics for given data list.
    Returns quality score and repetition ratio.
    """
    try:
        correct_count = 0
        wrong_count = 0
        repetition_count = 0
        total = len(data_list)
        
        if total == 0:
            return None, None
            
        for item in data_list:
            if item.get('judgment') == 'Correct':
                correct_count += 1
            if item.get('judgment') == 'Wrong':
                wrong_count += 1
                if item.get('error_type') == 'Repetition':
                    repetition_count += 1
        
        score = correct_count / total if total > 0 else None
        ratio = repetition_count / wrong_count if wrong_count > 0 else None
        
        return score, ratio
    except Exception:
        return None, None

def analyze_reflection(json_file_path):
    """
    Analyze reflection quality metrics for a single json file.
    Returns metrics with category information.
    """
    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Get category info
        category = data.get('category', None)
        subcategory = data.get('subcategory', None)
        question_type = data.get('question_type', None)
            
        json_str = extract_json_string(data.get('valid_outputs'))
        if not json_str:
            raise ValueError('Missing or invalid valid_outputs field')
            
        steps_data = json.loads(json_str, strict=False)
        if not steps_data:
            raise ValueError('Empty steps_data')
        
        overall_score, repetition_ratio = calculate_reflection_quality(steps_data)
        
        # Update the file with quality metrics
        data['reflection_quality'] = {
            'score': overall_score,
            'repetition_ratio': repetition_ratio
        }
        
        with open(json_file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        
        return {
            'success': True,
            'score': overall_score,
            'ratio': repetition_ratio,
            'category': category,
            'subcategory': subcategory,
            'question_type': question_type
        }
    except json.JSONDecodeError as e:
        return {'success': False, 'error': f'JSON decode error: {str(e)}'}
    except ValueError as e:
        return {'success': False, 'error': str(e)}
    except Exception as e:
        return {'success': False, 'error': f'Unknown error: {str(e)}'}

def process_model_files(model_path):
    """Process all files in model directory and aggregate results"""
    results = {
        'scores': [],
        'ratios': [],
        'error_files': [],
        'category_metrics': defaultdict(list),
        'subcategory_metrics': defaultdict(list),
        'question_type_metrics': defaultdict(list)
    }
    
    for json_file in os.listdir(model_path):
        if not json_file.endswith('.json'):
            continue
            
        file_path = os.path.join(model_path, json_file)
        result = analyze_reflection(file_path)
        
        if result.get('success'):
            if result['score'] is not None:
                results['scores'].append(result['score'])
            if result['ratio'] is not None:
                results['ratios'].append(result['ratio'])
                
            # Collect category metrics
            if result['category']:
                results['category_metrics'][result['category']].append(result['score'])
            if result['subcategory']:
                results['subcategory_metrics'][result['subcategory']].append(result['score'])
            if result['question_type']:
                results['question_type_metrics'][result['question_type']].append(result['score'])
        else:
            results['error_files'].append({
                'file': json_file,
                'error': result.get('error', 'Unknown error')
            })
    
    return results

def process_all_models(cache_dir, save_path):
    """Process all models and save aggregated results"""
    model_stats = {}
    results_data = {}
    all_error_files = {}
    
    # Create output directory
    save_dir = os.path.join(save_path, 'reflection_quality')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    
    for model in tqdm.tqdm(os.listdir(cache_dir)):
        model_path = os.path.join(cache_dir, model)
        if not os.path.isdir(model_path):
            continue
            
        results = process_model_files(model_path)
        model_stats[model] = results
        
        if results['error_files']:
            all_error_files[model] = results['error_files']
            
        # Calculate averages
        model_results = {
            "overall_metrics": {
                "average_reflection_quality": round(sum(results['scores'])/len(results['scores']), 4) if results['scores'] else None,
                "average_repetition_ratio": round(sum(results['ratios'])/len(results['ratios']), 4) if results['ratios'] else None
            },
            "category": {},
            "subcategory": {},
            "question_type": {}
        }
        
        # Add category metrics
        for cat, values in results['category_metrics'].items():
            non_empty_values = [val for val in values if val]
            model_results["category"][cat] = round(sum(non_empty_values)/len(non_empty_values), 4)
            
        # Add subcategory metrics
        for subcat, values in results['subcategory_metrics'].items():
            non_empty_values = [val for val in values if val]
            if non_empty_values:
                model_results["subcategory"][subcat] = round(sum(non_empty_values)/len(non_empty_values), 4)
            else:
                model_results["subcategory"][subcat] = None

        # Add question type metrics
        for qtype, values in results['question_type_metrics'].items():
            non_empty_values = [val for val in values if val]
            if non_empty_values:
                model_results["question_type"][qtype] = round(sum(non_empty_values)/len(non_empty_values), 4)
            else:
                model_results["question_type"][qtype] = None
        
        results_data[model] = model_results

    # Save main results
    output_file = os.path.join(save_dir, 'reflection_quality_results.json')
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    # Save error file if exists
    if all_error_files:
        error_file = os.path.join(save_dir, 'reflection_quality_errors.json')
        with open(error_file, 'w', encoding='utf-8') as f:
            json.dump(all_error_files, f, indent=4, ensure_ascii=False)

    return model_stats

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='calculate reflection quality')
    parser.add_argument('--cache_dir', type=str, 
                       default='./cache/reflection_quality',
                       help='cache directory')
    parser.add_argument('--save_path', type=str,
                       default='./final_results',
                       help='output directory')
    args = parser.parse_args()
    process_all_models(args.cache_dir, args.save_path)
    print('Done')