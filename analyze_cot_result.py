import os
import json
import glob

from analyze_scores import load_file
 
def extract_cot_result(data):
    results = {}
    for task, task_data in data.items():
        if task == "benchmeatinfo":
            continue
        results[task] = {}
        for example_num, example_data in task_data.items():
            if 'cot_stage1_output' in example_data:
                result = example_data['cot_stage1_output']
                results[task][example_num] = result
    return results

def process_text(text, char_per_line=100):
    return text
    for i, char in enumerate(text, start=1):
        if i % char_per_line == 0:
            if char == ' ':
                text = text[:i] + '\n' + text[i+1:]
            else:
                text = text[:i] + '-\n' + text[i:]
    return text

def main():
    gt_path = '/tmp2/cywu/VLABench/logs/gt/Complex_gt_operation_sequence.json'
    file_paths = [
        '/tmp2/cywu/VLABench/logs/vlm/Qwen2_VL/en/1_shot_CoT/Complex/output.json',
        '/tmp2/cywu/VLABench/logs/vlm/Qwen2_VL/en/1_shot_CoT_oracle/Complex/output.json'
    ]
    data = {}
    for file_path in file_paths:
        file_data = load_file(file_path)
        data_key = file_path.split('/')[-3]
        data[data_key] = extract_cot_result(file_data)
    gt_data = load_file(gt_path)
    results = {}
    for task, task_data in gt_data.items():
        results[task] = {}
        for example_num, example_data in task_data.items():
            results[task][example_num] = {}
            if 'instruction' in example_data:
                gt = example_data['instruction']
                results[task][example_num]['gt'] = process_text(gt)
            for key in data:
                pred = data[key].get(task, {}).get(example_num, None)
                results[task][example_num][key] = process_text(pred)
    with open('cot_analysis_result.json', 'w') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    main()
