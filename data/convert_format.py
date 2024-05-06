import json

tools = [{
    "name": "search_hotels",
    "description": "根据用户的需求生成查询条件来查酒店",
    "parameters": {
        "type": "object",
        "properties": {
            "name": { "type": "string", "description": "酒店名称" },
            "type": { "type": "string", "enum": ["豪华型", "经济型", "舒适型", "高档型"], "description": "酒店类型" },
            "facilities": { "type": "array", "items": { "type": "string" }, "description": "酒店能提供的设施列表" },
            "price_range_lower": { "type": "number", "minimum": 0, "description": "价格下限" },
            "price_range_upper": { "type": "number", "minimum": 0, "description": "价格上限" },
            "rating_range_lower": { "type": "number", "minimum": 0, "maximum": 5, "description": "评分下限" },
            "rating_range_upper": { "type": "number", "minimum": 0, "maximum": 5, "description": "评分上限" }
    }, "required": [] }
}]
tool_description = json.dumps(tools, ensure_ascii=False)

def read_jsonl(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

def write_jsonl(data, filename):
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            json_str = json.dumps(item,ensure_ascii=False)
            f.write(json_str + '\n')

def is_subset(sub_list, main_list):
    return all(item in main_list for item in sub_list)

def filter_subsets(lst):
    parsed_contexts = [(item, json.loads(item['context'])) for item in lst]
    return [item for item, context in parsed_contexts if not any(
        is_subset(context, json.loads(main_item['context'])) and item != main_item
        for main_item in lst)]

def convert(input_filename, output_filename):
    dataset = []
    lines = filter_subsets(read_jsonl(input_filename))
    for line in lines:
        data = {"tools":[tool_description.strip()],"conversations":[]}
        dialog = []
        dialog.extend(eval(line['context']))
        dialog.append(eval(line['response']))
        for turn in dialog:
            if turn["role"] == "search":
                think = {"role":"assistant","content":"我需要使用search_hotels工具来查询酒店"}
                data["conversations"].append(think)
                action = {"role":"tool","name":"search_hotels","parameters":turn["arguments"]}
                data["conversations"].append(action)
            elif turn["role"] == "return":
                data["conversations"][-1]["observation"] = turn["records"]
            else:
                data["conversations"].append(turn)
        dataset.append(data)
    write_jsonl(dataset, output_filename)

if __name__ == '__main__':
    convert('train.llama3.jsonl', 'train.chatglm3.jsonl')
    convert('dev.llama3.jsonl', 'dev.chatglm3.jsonl')
    convert('test.llama3.jsonl', 'test.chatglm3.jsonl')
