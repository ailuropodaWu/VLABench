import json
import os
import glob

def load_file(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} does not exist.")
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_scores(score_data, result_data):
    scores = {}
    for task, task_data in score_data.items():
        if task == "benchmeatinfo":
            continue
        error_count = 0
        task_scores = []
        exact_match_scores = []
        skill_match_scores = []
        entity_match_scores = []
        for example_num, example_data in task_data.items():
            if 'total_score' in example_data:
                task_scores.append(example_data['total_score'])
            if 'exact_match_score' in example_data:
                exact_match_scores.append(example_data['exact_match_score'])
            if 'skill_match_score' in example_data:
                skill_match_scores.append(example_data['skill_match_score'])
            if 'entity_match_score' in example_data:
                entity_match_scores.append(example_data['entity_match_score'])
            if 'format_error' in result_data[task][example_num]:
                error_count += 1

        scores[task] = {}
        if task_scores:
            scores[task]['total'] = sum(task_scores) / len(task_scores)
        if skill_match_scores:
            scores[task]['skill_match'] = sum(skill_match_scores) / len(skill_match_scores)
        if entity_match_scores:
            scores[task]['entity_match'] = sum(entity_match_scores) / len(entity_match_scores)
        if exact_match_scores:
            scores[task]['exact_match'] = sum(exact_match_scores) / len(exact_match_scores)
            scores[task]['exact_match_cnt'] = sum(100. for score in exact_match_scores if score == 100) / len(exact_match_scores)
        scores[task]['format_error'] = error_count * 100. / len(task_data) if task_data else 0
    summarize_score = {}
    summarize_score['total'] = sum(scores[task]['total'] for task in scores) / len(scores) if scores else 0
    summarize_score['skill_match'] = sum(scores[task]['skill_match'] for task in scores) / len(scores) if scores else 0
    summarize_score['entity_match'] = sum(scores[task]['entity_match'] for task in scores) / len(scores) if scores else 0
    summarize_score['exact_match'] = sum(scores[task]['exact_match'] for task in scores) / len(scores) if scores else 0
    summarize_score['exact_match_cnt'] = sum(scores[task]['exact_match_cnt'] for task in scores) / len(scores) if scores else 0
    summarize_score['format_error'] = sum(scores[task]['format_error'] for task in scores) / len(scores) if scores else 0
    scores['total'] = summarize_score
    # print(scores)
    return scores

def get_score_data_key(path):
    path = path.replace(os.getenv('VLABENCH_ROOT') + '/../logs/', '').replace('/final_score.json', '')
    path = path.replace('/', '_')
    path = path.replace('vlm_Qwen2_VL_', '').replace('en_', '').replace('_Complex', '')
    return path

def main():
    score_path_regex = os.path.join(os.getenv('VLABENCH_ROOT'), '../logs', '**/final_score.json')
    all_score_pathes = glob.glob(score_path_regex, recursive=True)
    all_data = {}
    # print(all_score_pathes)
    eval_dimension = ["M&T", "CommonSense", "Semantic", "Spatial", "PhysicalLaw", "Complex"]
    for score_path in all_score_pathes:
        if not "Complex" in score_path:
            continue
        # if "max_tok_" in score_path:
        #     continue
        result_path = score_path.replace('final_score.json', 'output.json')
        score = load_file(score_path)
        result = load_file(result_path)
        data_key = get_score_data_key(score_path)
        try:
            all_data[data_key] = extract_scores(score, result)
        except KeyError:
            print(f"Error processing {score_path}, skipping...")
    # Get all task names from the first data entry
    first_key = next(iter(all_data.keys()))
    task_names = [task for task in all_data[first_key].keys()]
    
    # Calculate appropriate column widths
    method_width = max(len(key) for key in all_data.keys()) + 2
    method_width = max(method_width, 25)  # Minimum width for readability
    
    # Wider columns for score data to prevent wrapping
    score_width = 35  # Increased from 25 to accommodate longer score strings
    
    with open(os.path.join(os.getenv('VLABENCH_ROOT'), '../logs/score_summary.txt'), 'w') as f:
        # Write simple header without complex formatting
        f.write("Method".ljust(method_width))
        for task in task_names:
            f.write(task.ljust(score_width))
        f.write("\n")
        
        # Write sub-header
        f.write("score/skill/entity/exact/cnt/error%".ljust(method_width))
        for _ in task_names:
            f.write("total/skill/entity/exact/cnt/error%".ljust(score_width))
        f.write("\n")
        
        # Write separator line
        total_width = method_width + score_width * len(task_names)
        f.write("-" * total_width + "\n")

        # Write data rows (simple single-line format)
        for key, data in all_data.items():
            # Write method name (truncate if too long)
            if len(key) > method_width - 1:
                f.write(key[:method_width-4] + "...")
            else:
                f.write(key.ljust(method_width))
            
            # Write scores for each task
            for task in task_names:
                score = data.get(task, {})
                if score:
                    # Simplified format with shorter precision
                    value_text = f"{score['total']:.1f}/{score['skill_match']:.1f}/{score['entity_match']:.1f}/{score['exact_match']:.1f}/{score['exact_match_cnt']:.2f}/{score['format_error']:.1f}%"
                else:
                    value_text = "N/A"
                
                # Truncate if still too long
                if len(value_text) > score_width - 1:
                    value_text = value_text[:score_width-4] + "..."
                
                f.write(value_text.ljust(score_width))
            f.write("\n")
        
        # Final separator
        f.write("-" * total_width + "\n")

if __name__ == "__main__":
    main()