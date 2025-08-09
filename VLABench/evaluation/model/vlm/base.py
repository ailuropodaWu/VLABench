import json
import os
import torch

class BaseVLM():
    def __init__(self) -> None:
        self.name =self.get_name()

    def evaluate(self, input_dict, language, with_CoT=False, task_to_have_oracle=None):
        raise NotImplementedError
    
    def get_name(self):
        return "BaseVLM"
    
def get_ti_list(input_dict, language, with_CoT=False, task_to_have_oracle=None):
    if language == "zh":
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])
        if "shot_input_pic" in input_dict:
            for shot_num in input_dict["shot_input_pic"]:
                ti_list.append(["text", "示范"+shot_num+"输入图片:" ])
                ti_list.append(["image", input_dict["shot_input_pic"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"输入带编号标签的图片" ])
                ti_list.append(["image", input_dict["shot_input_pic_gt"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"语言指令:"])
                ti_list.append(["text", input_dict["shot_input_instruction"][shot_num]])
                ti_list.append(["text", "示范"+shot_num+"输出技能序列"])
                ti_list.append(["text", json.dumps(input_dict["shot_output"][shot_num])])
        ti_list.append(["text", "输入图片" ])
        ti_list.append(["image", input_dict["input_pic"]])
        ti_list.append(["text", "输入带编号标签的图片" ])
        ti_list.append(["image", input_dict["input_pic_gt"]])
        ti_list.append(["text", "语言指令:"])
        ti_list.append(["text", input_dict["input_instruction"]])
        ti_list.append(["text", "请你给出输出的技能序列"])
        if with_CoT:
            ti_list.append(["text", "请一步一步分析问题最后给出答案"])
    elif language == "en":
        ti_list = []
        ti_list.append(["text", input_dict["pre_prompt"] ])
        if "shot_input_pic" in input_dict:
            for shot_num in input_dict["shot_input_pic"]:
                ti_list.append(["text", "Example "+shot_num+" input picture:" ])
                ti_list.append(["image", input_dict["shot_input_pic"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" input picture with numbered tags" ])
                ti_list.append(["image", input_dict["shot_input_pic_gt"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" language instruction:"])
                ti_list.append(["text", input_dict["shot_input_instruction"][shot_num]])
                ti_list.append(["text", "Example "+shot_num+" output skill sequence"])
                ti_list.append(["text", json.dumps(input_dict["shot_output"][shot_num])])
        ti_list.append(["text", "Input picture" ])
        ti_list.append(["image", input_dict["input_pic"]])
        ti_list.append(["text", "Input picture with numbered tags" ])
        ti_list.append(["image", input_dict["input_pic_gt"]])
        ti_list.append(["text", "Language instruction:"])
        ti_list.append(["text", input_dict["input_instruction"]])
        if task_to_have_oracle is not None:
            try:
                oracle_prompt = prepare_oracle_promt(task_to_have_oracle, input_dict["input_instruction"])
                ti_list.append(["text", "Please refer to following information to finish the task: " + oracle_prompt])
            except ValueError as e:
                pass
        ti_list.append(["text", "Please give the output skill sequence"])
        if with_CoT:
            ti_list.append(["text", "Please analyze the problem step by step and give the answer"])
    else:
        raise ValueError("language should be zh or en")
    return ti_list

def prepare_oracle_promt(task_name, instruction):
    if task_name == "cook_dishes":
        recipe_config_path = os.path.join(os.getenv("VLABENCH_ROOT"), "configs", "task_related", "recipe.json")
        recipe_config = json.load(open(recipe_config_path)) # {"train": {"name": [...]}, "eval": {"name": [...]}}
        all_recipe = {**recipe_config["train"], **recipe_config["eval"]}
        for dish_name, ingredients in all_recipe.items():
            if dish_name in instruction:
                return f"To cook {dish_name}, you need the following ingredients: {', '.join(ingredients)}."
        raise ValueError(f"Dish name not found in instruction.\nInstruction: {instruction}")
    elif task_name == "take_chemistry_experiment":
        experiment_config_path = os.path.join(os.getenv("VLABENCH_ROOT"), "configs", "task_related", "experiment.json")
        experiment_config = json.load(open(experiment_config_path)) # {"experiment_i": {"instruction": '...', "solutions": [...]}}
        for experiment_info in experiment_config.values():
            if experiment_info["instruction"] in instruction:
                return f"To conduct the experiment, you need the following materials: {', '.join(experiment_info['solutions'])}."
        raise ValueError(f"Experiment instruction not found in instruction.\nInstruction: {instruction}")
    elif task_name == "texas_holdem":
        return (
            "In Texas Hold'em, the hand rankings from low to high are:" 
            "Single Card" 
            "One Pair: two cards of the same rank" 
            "Two Pair: two cards of one rank and two cards of another rank" 
            "Three of a Kind: three cards of the same rank" 
            "Straight: five consecutive cards of different suits" 
            "Flush: five cards of the same suit" 
            "Full House: three cards of one rank and two cards of another rank" 
            "Four of a Kind: four cards of the same rank" 
            "Straight Flush: five consecutive cards of the same suit" 
            "Royal Flush: the highest straight flush."
        )
    elif task_name == "book_rearrange":
        return "Order the book according to the publish date."
    elif task_name == "hammer_nail_and_hang_picture":
        return "The nail should be hammered into the wall before the picture is hung."
    else:
        raise ValueError(f"Unknown task name: {task_name}")
