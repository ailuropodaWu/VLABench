import json
import os
import glob

def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_scores(score_data):
    scores = {}
    for task, task_data in score_data.items():
        task_scores = []
        for example_num, example_data in task_data.items():
            if 'total_score' in example_data:
                task_scores.append(example_data['total_score'])
        if task_scores:
            scores[task] = sum(task_scores) / len(task_scores)
        else:
            print(f"Warning: No final score found for task {task}.")
    scores['total_score'] = sum(scores.values()) / len(scores) if scores else 0
    return scores

def extract_format_error(result_data):
    format_errors = {}
    for task, task_data in result_data.items():
        if task == "benchmeatinfo":
            continue
        error_count = 0
        for example_num, example_data in task_data.items():
            if 'format_error' in example_data:
                error_count += 1
        format_errors[task] = error_count * 100. / len(task_data) if task_data else 0
    format_errors['total_format_error'] = sum(format_errors.values()) / len(format_errors) if format_errors else 0
    return format_errors

def get_score_data_key(path):
    path = path.replace(os.getenv('VLABENCH_ROOT') + '/../logs/', '').replace('/final_score.json', '')
    path = path.replace('/', '_')
    return path

def main():
    score_path_regex = os.path.join(os.getenv('VLABENCH_ROOT'), '../logs', '**/final_score.json')
    all_score_pathes = glob.glob(score_path_regex, recursive=True)
    all_data = {}
    # print(all_score_pathes)
    eval_dimension = ["M&T", "CommonSense", "Semantic", "Spatial", "PhysicalLaw", "Complex"]
    for score_path in all_score_pathes:
        result_path = score_path.replace('final_score.json', 'output.json')
        score = load_file(score_path)
        result = load_file(result_path)
        data_key = get_score_data_key(score_path)
        all_data[data_key] = {
            'scores': extract_scores(score),
            'format_errors': extract_format_error(result)
        }
    with open(os.path.join(os.getenv('VLABENCH_ROOT'), '../logs/score_summary.txt'), 'w') as f:
        for key, data in all_data.items():
            f.write(f"Scores for {key}:\n")
            for task, task_score in data['scores'].items():
                f.write(f"  {task} - score: {task_score:.2f} format error {data['format_errors'].get(task, 0):.2f}\n")

if __name__ == "__main__":
    main()