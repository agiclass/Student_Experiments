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
            prompt += f'<|im_start|>{turn["role"]}\n{turn["content"]}<|im_end|>\n'
        else:
            if turn["role"] == "search":
                obj = turn["arguments"]
                filtered_obj = {k: v for k, v in obj.items() if v is not None}
                prompt += '<|im_start|>search\n'
                prompt += json.dumps(filtered_obj,indent=4,ensure_ascii=False)
            else:
                obj = turn["records"]
                prompt += '<|im_start|>return\n'
                prompt += json.dumps(obj,indent=4,ensure_ascii=False)
            prompt += '<|im_end|>\n'
    return prompt

def build_response(response):
    if isinstance(response,str):
        response = json.loads(response)
    if response["role"] == "assistant":
        return '<|im_start|>assistant\n' + response["content"] + '<|im_end|>'
    else:
        obj = response["arguments"]
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        return '<|im_start|>search\n' + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + '<|im_end|>'

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
