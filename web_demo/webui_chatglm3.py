import json
import gradio as gr
import pandas as pd
from copy import deepcopy
from db_client import HotelDB

import sys
sys.path.append('../chatglm3')
from cli_evaluate import load_model, load_lora, load_pt2
from arguments import ModelArguments, PeftArguments
from transformers import HfArgumentParser

parser = HfArgumentParser((ModelArguments, PeftArguments))
model_args, peft_args = parser.parse_args_into_dataclasses()

db = HotelDB()
tokenizer, model = None, None

if model_args.checkpoint_path:
    if 'pt2' in model_args.checkpoint_path:
        tokenizer, model = load_pt2(model_args)
    elif 'lora' in model_args.checkpoint_path:
        tokenizer, model = load_lora(model_args, peft_args)
else:
    tokenizer, model = load_model(model_args)

system_prompt = 'Answer the following questions as best as you can. You have access to the following tools. Carefully check the return result of the function before answering user\'s question'
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

def chat(query, history, role):
    eos_token_id = [tokenizer.eos_token_id, 
                    tokenizer.get_command("<|user|>"), 
                    tokenizer.get_command("<|observation|>")]
    inputs = tokenizer.build_chat_input(query, history=history, role=role)
    inputs = inputs.to('cuda')
    outputs = model.generate(**inputs, max_length=4096, eos_token_id=eos_token_id)
    outputs = outputs.tolist()[0][len(inputs["input_ids"][0]):-1]
    response = tokenizer.decode(outputs)
    history.append({"role": role, "content": query})
    history = deepcopy(history)
    for response in response.split("<|assistant|>"):
        splited = response.split("\n", maxsplit=1)
        if len(splited) == 2:
            metadata, response = splited
        else:
            metadata = ""
            response = splited[0]
        if not metadata.strip():
            response = response.strip()
            history.append({"role": "assistant", "metadata": metadata, "content": response})
        else:
            history.append({"role": "assistant", "metadata": metadata, "content": response})
            response = "\n".join(response.split("\n")[1:-1])
            def tool_call(**kwargs):
                return kwargs
            try:
                parameters = eval(response)
            except:
                parameters = {}
            response = {"name": metadata.strip(), "parameters": parameters}
    return response, history

def handler(user_input, chatbot, history, prompt_field, tools_field, search_field, return_field):
    history[0] = build_history(prompt_field, tools_field)[0]
    response, history = chat(user_input, history, 'user')
    if isinstance(response, dict):
        parameters = response['parameters']
        search_field = json.dumps(parameters,indent=4,ensure_ascii=False)
        return_field = db.search(parameters, limit=3)
        observation = json.dumps(return_field, ensure_ascii=False)
        response, history = chat(observation, history, 'observation')
        reply = response.strip()
        keys = []
        if return_field:
            keys = ['name', 'address', 'phone', 'price', 'rating', 'subway', 'type', 'facilities']
        data = {key: [item[key] for item in return_field] for key in keys}
        data = data or {"hotel": []}
        return_field = pd.DataFrame(data)
    elif isinstance(response, str):
        reply = response.strip()
    chatbot.append((user_input, reply))
    return "", chatbot, history, search_field, return_field

def build_history(system_prompt, tool_description):
    default_history = [{'role':'system','content':f'{system_prompt}:\n[\n{tool_description}\n]'}]
    return default_history

def reset_state():
    return "", [], build_history(system_prompt, tool_description), system_prompt, tool_description, "", None

def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Hotel Chatbot</h1>""")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=1)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    resetBtn = gr.Button("清空")
            with gr.Column(scale=2):
                gr.HTML("""<h4>system prompt</h4>""")
                prompt_field = gr.Textbox(show_label=False, interactive=True, value=system_prompt, lines=1)
                gr.HTML("""<h4>tool description</h4>""")
                tools_field = gr.Textbox(show_label=False, interactive=True, value=tool_description, lines=4)
                gr.HTML("""<h4>function parameter</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="搜索条件...", lines=6)

        with gr.Row():
            with gr.Column():
                return_field = gr.Dataframe()

        history = gr.State(build_history(system_prompt, tool_description))

        submitBtn.click(handler, [user_input, chatbot, history, prompt_field, tools_field, search_field, return_field],
                        [user_input, chatbot, history, search_field, return_field])
        resetBtn.click(reset_state, outputs=[user_input, chatbot, history, prompt_field, tools_field, search_field, return_field])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=6006, inbrowser=True)

if __name__ == "__main__":
    main()
