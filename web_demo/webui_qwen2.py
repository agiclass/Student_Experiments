import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.append('../qwen2')
import json
import torch
import argparse
import gradio as gr
import pandas as pd
from db_client import HotelDB
from evaluate import load_model
from data_preprocess import build_prompt, parse_json

# init gloab variables
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
parser.add_argument("--ckpt", type=str, default=None, required=True, help="The checkpoint path")
args = parser.parse_args()

db = HotelDB()
tokenizer, model = load_model(args.model, args.ckpt)

def get_completion(prompt):
    inputs = tokenizer([prompt], return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=1024)
        response = tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
    return response

def remove_search_history(context):
    i = 0
    while i < len(context):
        if context[i]['role'] in ['search','return']:
            del context[i]
        else:
            i += 1

def chat(user_input, chatbot, context, search_field, return_field):
    context.append({'role':'user','content':user_input})
    response = get_completion(build_prompt(context))
    #print(response)
    # 判断以search命令开头时去执行搜索
    if "search" in response:
        # 取出最新一条 'search' 后面的json查询条件
        search_query = parse_json(response)
        if search_query is not None:
            search_field = json.dumps(search_query,indent=4,ensure_ascii=False)
            remove_search_history(context)
            context.append({'role':'search','arguments':search_query})
            # 调用酒店查询接口
            return_field = db.search(search_query, limit=3)
            context.append({'role':'return','records':return_field})
            keys = []
            if return_field:
                keys = ['name', 'address', 'phone', 'price', 'rating', 'subway', 'type', 'facilities']
            data = {key: [item[key] for item in return_field] for key in keys}
            data = data or {"hotel": []}
            return_field = pd.DataFrame(data)
            # 将查询结果发给LLM，再次那么让LLM生成回复
            response = get_completion(build_prompt(context))

    reply = response.replace("assistant", "")
    chatbot.append((user_input, reply))
    context.append({'role':'assistant','content':reply})
    return "", chatbot, context, search_field, return_field

def reset_state():
    return [], [], "", "", None

def main():
    with gr.Blocks() as demo:
        gr.HTML("""<h1 align="center">Hotel Chatbot (Qwen2 LoRA)</h1>""")

        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
            with gr.Column(scale=2):
                gr.HTML("""<h4>Search</h4>""")
                search_field = gr.Textbox(show_label=False, placeholder="search...", lines=8)
                user_input = gr.Textbox(show_label=False, placeholder="输入框...", lines=2)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                    emptyBtn = gr.Button("清空")

        with gr.Row():
            with gr.Column():
                gr.HTML("""<h4>Return</h4>""")
                return_field = gr.Dataframe()

        context = gr.State([])

        submitBtn.click(chat, [user_input, chatbot, context, search_field, return_field],
                        [user_input, chatbot, context, search_field, return_field])
        emptyBtn.click(reset_state, outputs=[chatbot, context, user_input, search_field, return_field])

    demo.queue().launch(share=False, server_name='0.0.0.0', server_port=6006, inbrowser=True)

if __name__ == "__main__":
    main()
