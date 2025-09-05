from VLABench.evaluation.model.vlm.base import *

class Qwen2_VL_3b(BaseVLM):
    def __init__(self) -> None:
        super().__init__()
        
        from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
        from modelscope import snapshot_download
        model_dir = snapshot_download("qwen/Qwen2-VL-3B-Instruct")

        # # default: Load the model on the available device(s)
        # model = Qwen2VLForConditionalGeneration.from_pretrained(
        #     model_dir, torch_dtype="auto", device_map="auto"
        # )

        # We recommend enabling flash_attention_2 for better acceleration and memory saving, especially in multi-image and video scenarios.
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_dir,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        ).cuda()

        # default processer
        self.processor = AutoProcessor.from_pretrained(model_dir)

    def get_response(self, ti_list):
        from qwen_vl_utils import process_vision_info
        content = self.build_prompt_with_tilist(ti_list)
        # Messages containing multiple images and a text query
        messages = [
            {
                "role": "user",
                "content": content,
            }
        ]

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference
        generated_ids = self.model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        return output_text[0]


    def build_prompt_with_tilist(self, ti_list):
        content = []
        for ti in ti_list:
            if ti[0] == "text":
                content.append({"type": "text", "text": ti[1]})
            elif ti[0] == "image":
                content.append({"type": "image", "image": ti[1]})
        return content
    
    def get_name(self):
        return "Qwen2_VL"