import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../glm4')
import json
import argparse
import gradio as gr
import pandas as pd
from db_client import HotelDB
from evaluate import load_model, load_raw_model, get_completion

# init gloab variables
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
parser.add_argument("--ckpt", type=str, default=None, required=False, help="The checkpoint path")
args = parser.parse_args()

system_prompt = {"role": "system", "content": "", "tools": [{"type": "function", "function": {"name": "search_hotels", "description": "根据用户的需求生成查询条件来查酒店", "parameters": {"type": "object", "properties": {"name": {"type": "string", "description": "酒店名称"}, "type": {"type": "string", "enum": ["豪华型", "经济型", "舒适型", "高档型"], "description": "酒店类型"}, "facilities": {"type": "array", "items": {"type": "string"}, "description": "酒店能提供的设施列表"}, "price_range_lower": {"type": "number", "minimum": 0, "description": "价格下限"}, "price_range_upper": {"type": "number", "minimum": 0, "description": "价格上限"}, "rating_range_lower": {"type": "number", "minimum": 0, "maximum": 5, "description": "评分下限"}, "rating_range_upper": {"type": "number", "minimum": 0, "maximum": 5, "description": "评分上限"}}, "required": []}}}]}

db = HotelDB()
if args.ckpt:
    tokenizer, model = load_model(args.model, args.ckpt)
else:
    tokenizer, model = load_raw_model(args.model)

def chat(user_input, chatbot, context, search_field, return_field):
    context.append({'role':'user','content':user_input})
    role, end, content = get_completion(tokenizer, model, context)
    assert role == '<|assistant|>'
    content = content.strip()
    context.append({'role':'assistant','content':content})
    if end == '<|observation|>' or content.startswith('search_hotels'):
        try:
            _, arguments = content.split('\n')
        except:
            _, arguments = '','{}'
        arguments = json.loads(arguments)
        return_field = db.search(arguments, limit=3)
        context.append({'role':'observation','content':json.dumps(return_field,ensure_ascii=False)})
        keys = []
        if return_field:
            keys = ['name', 'address', 'phone', 'price', 'rating', 'subway', 'type', 'facilities']
        data = {key: [item[key] for item in return_field] for key in keys}
        data = data or {"hotel": []}
        return_field = pd.DataFrame(data)
        _, _, response = get_completion(tokenizer, model, context)
    elif end == '<|user|>' or end == '<|endoftext|>':
        response = content
        arguments = None
    if arguments:
        search_field = json.dumps(arguments,indent=4,ensure_ascii=False)
    else:
        search_field = ""
    chatbot.append((user_input, response.strip()))
    context.append({'role':'assistant','content':response.strip()})
    return "", chatbot, context, search_field, return_field

def reset_state():
    return [], [], "", "", None

def main():
    with gr.Blocks() as demo:
        if args.ckpt:
            gr.HTML("""<h1 align="center">Hotel Chatbot (GLM-4 QLoRA)</h1>""")
        else:
            gr.HTML("""<h1 align="center">Hotel Chatbot (GLM-4 Original)</h1>""")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=1)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")
            with gr.Column(scale=2):
                gr.HTML("""<h4>System Prompt</h4>""")
                system_prompt_field = gr.Textbox(show_label=False, interactive=True, 
                        value=json.dumps(system_prompt, ensure_ascii=False), lines=1)
                gr.HTML("""<h4>Search</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="搜索条件...", lines=6)

        with gr.Row():
            with gr.Column():
                return_field = gr.Dataframe()

        context = gr.State([system_prompt])

        submitBtn.click(chat, [user_input, chatbot, context, search_field, return_field],
                        [user_input, chatbot, context, search_field, return_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field, return_field])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=6006, inbrowser=True)

if __name__ == "__main__":
    main()
