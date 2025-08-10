import os
import json

def load_single_output(data_path):
    gt_operation_sequence_path = os.path.join(data_path, 'output/operation_sequence.json')
    with open(gt_operation_sequence_path) as f:
        gt_operation_sequence = json.load(f)
    instruction_file_path = os.path.join(data_path, 'input/instruction.txt')
    gt_operation_sequence['instruction'] = open(instruction_file_path, 'r').read().strip()
    return gt_operation_sequence

eval_dimension = ["M&T", "CommonSense", "Semantic", "Spatial", "PhysicalLaw", "Complex"]
for eval_dim in eval_dimension:
    output_path = os.path.join(os.getenv("VLABENCH_ROOT"), "../logs/gt", f"{eval_dim}_gt_operation_sequence.json")
    if os.path.exists(output_path):
        print(f"Output file {output_path} already exists. Skipping extraction.")
        continue
    all_gt_operation_sequences = {}
    task_list = os.listdir(os.path.join(os.getenv("VLABENCH_ROOT"), "../dataset", f"vlm_evaluation_v1.0/{eval_dim}"))
    for task_name in task_list:
        task_path = os.path.join(os.getenv("VLABENCH_ROOT"), "../dataset", f"vlm_evaluation_v1.0/{eval_dim}", task_name)
        if not os.path.isdir(task_path):
            continue
        example_num = 0
        while True:
            example_path = os.path.join(task_path, "example" + str(example_num))
            if not os.path.exists(example_path):
                break
            gt_operation_sequence = load_single_output(example_path)
            if task_name not in all_gt_operation_sequences:
                all_gt_operation_sequences[task_name] = {}
            all_gt_operation_sequences[task_name][f"{example_num}"] = gt_operation_sequence
            example_num += 1
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_gt_operation_sequences, f, indent=4)