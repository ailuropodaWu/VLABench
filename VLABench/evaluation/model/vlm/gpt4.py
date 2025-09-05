from VLABench.evaluation.model.vlm.base import *

class GPT_4v(BaseVLM):
    def __init__(self, api_key=None, base_url=None) -> None:
        self.api_key = api_key
        self.base_url = base_url
        super().__init__()

    def get_response(self, ti_list):
        from VLABench.utils.gpt_utils import build_prompt_with_tilist, query_gpt4_v
        prompt = build_prompt_with_tilist(ti_list)
        content = query_gpt4_v(prompt, api_key=self.api_key, base_url=self.base_url, model="gpt-4o")
        return content

    def get_name(self):
        return "GPT_4v"