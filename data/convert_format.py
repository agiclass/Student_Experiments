import json

tools = [{
    "type": "function",
    "function": {
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
    }
}]

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
        messages = [{"role":"system","content":"","tools":tools}]
        dialog = []
        dialog.extend(eval(line['context']))
        dialog.append(eval(line['response']))
        for turn in dialog:
            if turn["role"] == "search":
                content = "search_hotels\n"+json.dumps(turn["arguments"],ensure_ascii=False)
                messages.append({'role':'assistant','content':content})
            elif turn["role"] == "return":
                content = json.dumps(turn["records"], ensure_ascii=False)
                messages.append({'role':'observation','content':content})
            else:
                messages.append(turn)
        dataset.append({"messages":messages})
        tuples = [(item, len(json.dumps(item,ensure_ascii=False))) for item in dataset]
        sorted_tuple = sorted(tuples, key=lambda x: x[1])
        sorted_dataset = [item[0] for item in sorted_tuple]
        sorted_len = [item[1] for item in sorted_tuple]
        sorted_dataset = sorted_dataset[:sum(1 for num in sorted_len if num <= 3400)]
    write_jsonl(sorted_dataset, output_filename)

if __name__ == '__main__':
    convert('train.llama3.jsonl', 'train.glm4.jsonl')
    convert('dev.llama3.jsonl', 'dev.glm4.jsonl')
    convert('test.llama3.jsonl', 'test.glm4.jsonl')
