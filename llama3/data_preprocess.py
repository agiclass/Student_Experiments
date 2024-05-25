import json
from torch.utils.data import Dataset

class InputOutputDataset(Dataset):
    def __init__(self, data, tokenizer, args):
        super(InputOutputDataset, self).__init__()
        self.data = data
        self.tokenizer = tokenizer
        self.prompt_column = args.prompt_column
        self.response_column = args.response_column
        self.max_source_length = args.max_source_length
        self.max_target_length = args.max_target_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        item = self.data[i]
        # add_special_tokens 不在开头加 special_tokens
        context = self.tokenizer(
            build_prompt(item[self.prompt_column]), 
            max_length=self.max_source_length, 
            add_special_tokens=False)
        response = self.tokenizer(
            build_response(item[self.response_column]), 
            max_length=self.max_target_length, 
            add_special_tokens=False)
        input_ids = context["input_ids"] + response["input_ids"]
        attention_mask = context["attention_mask"] + response["attention_mask"]
        labels = [-100] * len(context["input_ids"]) + response["input_ids"]
        assert len(input_ids) == len(labels), f"length mismatch: {len(input_ids)} vs {len(labels)}"
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }

def build_prompt(context):
    if isinstance(context,str):
        context = json.loads(context)
    prompt = ''
    for turn in context:
        if turn["role"] in ["user","assistant"]:
            prompt += f'<|start_header_id|>{turn["role"]}<|end_header_id|>\n{turn["content"]}<|eot_id|>\n'
        else:
            if turn["role"] == "search":
                obj = turn["arguments"]
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                prompt += '<|start_header_id|>search<|end_header_id|>\n'
                prompt += json.dumps(filtered_obj,indent=4,ensure_ascii=False)
            else:
                obj = turn["records"]
                prompt += '<|start_header_id|>return<|end_header_id|>\n'
                prompt += json.dumps(obj,indent=4,ensure_ascii=False)
            prompt += '<|eot_id|>\n'
    return prompt

def build_response(response):
    if isinstance(response,str):
        response = json.loads(response)
    if response["role"] == "assistant":
        return '<|start_header_id|>assistant<|end_header_id|>\n' + response["content"] + '<|eot_id|>'
    else:
        obj = response["arguments"]
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        return '<|start_header_id|>search<|end_header_id|>\n' + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + '<|eot_id|>'

def parse_json(string):
    search_pos = 0
    # 开始寻找第一个 '{'
    start = string.find('{', search_pos)
    if start == -1:
        return None
    # 从找到的 '{' 位置开始，向后寻找最后一个 '}'
    end = string.rfind('}', start)
    if end == -1:
        return None
    # 提取并尝试解析 JSON
    json_string = string[start:end + 1]
    try:
        obj = json.loads(json_string)
        return obj
    except json.JSONDecodeError:
        return None
