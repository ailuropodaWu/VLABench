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
    path = path.replace('vlm_Qwen2_VL_', '').replace('_en_', '_')
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
    # Get all task names from the first data entry
    first_key = next(iter(all_data.keys()))
    task_names = [task for task in all_data[first_key]['scores'].keys()]
    fix_length = max(len(key) for key in all_data.keys()) + 5  # Add some padding for readability
    
    def format_text(text, width):
        """Format text to fit within width, splitting to two lines if necessary"""
        if len(text) <= width:
            return [text.ljust(width)]
        else:
            # Split at roughly half length, preferring word boundaries
            mid = width // 2
            split_point = mid
            # Try to find a good split point (underscore, space, or other separator)
            for i in range(mid - 5, mid + 5):
                if i < len(text) and text[i] in ['_', '-', ' ']:
                    split_point = i
                    break
            line1 = text[:split_point].ljust(width)
            line2 = text[split_point:].ljust(width)
            return [line1, line2]
    
    with open(os.path.join(os.getenv('VLABENCH_ROOT'), '../logs/score_summary.txt'), 'w') as f:
        # Format headers
        method_header = format_text("Method", fix_length)
        task_headers = [format_text(task, 25) for task in task_names]
        
        # Write header lines
        for line_idx in range(max(len(method_header), max(len(header) for header in task_headers))):
            # Write method header line
            if line_idx < len(method_header):
                f.write(method_header[line_idx])
            else:
                f.write(" " * fix_length)
            
            # Write task header lines
            for header in task_headers:
                if line_idx < len(header):
                    f.write(header[line_idx])
                else:
                    f.write(" " * 25)
            f.write("\n")
        
        f.write("-" * (fix_length + 25 * len(task_names)) + "\n")

        # Write data rows
        for key, data in all_data.items():
            if not "Complex" in key:
                continue
            if "max_token_128" in key:
                continue
            
            # Format key
            key_lines = format_text(key, fix_length)
            
            # Prepare value lines for each task
            value_lines = []
            for task in task_names:
                score = data['scores'].get(task, 0)
                format_error = data['format_errors'].get(task, 0)
                value_text = f"{score:.2f} / {format_error:.1f}%"
                value_lines.append(format_text(value_text, 25))
            
            # Write all lines for this row
            max_lines = max(len(key_lines), max(len(vl) for vl in value_lines))
            for line_idx in range(max_lines):
                # Write key line
                if line_idx < len(key_lines):
                    f.write(key_lines[line_idx])
                else:
                    f.write(" " * fix_length)
                
                # Write value lines
                for value_line in value_lines:
                    if line_idx < len(value_line):
                        f.write(value_line[line_idx])
                    else:
                        f.write(" " * 25)
                f.write("\n")

if __name__ == "__main__":
    main()