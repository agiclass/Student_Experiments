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
        input_ids = context["input_ids"] + response["input_ids"] + [self.tokenizer.pad_token_id]
        # 因为eos token也是要关注的所以补充为1
        attention_mask = context["attention_mask"] + response["attention_mask"] + [1]  
        labels = [-100] * len(context["input_ids"]) + response["input_ids"] + [self.tokenizer.pad_token_id]
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
                prompt += '<|start_header_id|>assistant<|end_header_id|>\n'
                prompt += turn["role"] + ":\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False)
            else:
                obj = turn["records"]
                prompt += '<|start_header_id|>user<|end_header_id|>\n'
                prompt += turn["role"] + ":\n" + json.dumps(obj,indent=4,ensure_ascii=False)   
            prompt += '<|eot_id|>\n'
    prompt += '<|start_header_id|>assistant<|end_header_id|>\n'
    return prompt

def build_response(response):
    if isinstance(response,str):
        response = json.loads(response)
    if response["role"] == "assistant":
        return response["content"] + '<|eot_id|>'
    else:
        obj = response["arguments"]
        filtered_obj = {k: v for k, v in obj.items() if v is not None}
        return "search:\n" + json.dumps(filtered_obj,indent=4,ensure_ascii=False) + '<|eot_id|>'
