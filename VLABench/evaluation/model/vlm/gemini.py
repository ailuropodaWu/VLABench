from VLABench.evaluation.model.vlm.base import *

class Gemini(BaseVLM):
    def __init__(self, api_key=None, base_url=None, model="gemini-2.5-pro-exp-03-25") -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        super().__init__()

    def get_response(self, ti_list):
        from VLABench.utils.gpt_utils import build_prompt_with_tilist, query_gpt4_v
        prompt = build_prompt_with_tilist(ti_list)
        content = query_gpt4_v(prompt, api_key=self.api_key, base_url=self.base_url, model=self.model)
        return content

    def get_name(self):
        return "Gemini"