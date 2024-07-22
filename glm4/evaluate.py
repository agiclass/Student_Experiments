import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def load_model(model_path, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        quantization_config=bnb_config,
        max_memory = {0:'24500MB'},
        trust_remote_code=True,
        empty_init=False,
        use_cache=False,
    )
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path)
    return tokenizer, model

def load_raw_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to("cuda").eval()
    return tokenizer, model

def get_completion(tokenizer, model, messages):
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")
    gen_kwargs = {"max_length": 3072, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        try:
            role = tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]-1][0], skip_special_tokens=False)
            end = tokenizer.decode(outputs[:,-1][0], skip_special_tokens=False)
            content = tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
        except:
            role, end, content = "", "", ""
    return role, end, content

def remove_empty(d):
    if type(d) is dict:
        return dict((k, remove_empty(v)) for k, v in d.items() if v and remove_empty(v))
    elif type(d) is list:
        return [remove_empty(v) for v in d if v and remove_empty(v)]
    else:
        return d

class Evaluator:
    """
    计算slot和reply业务指标
    """
    def __init__(self, tokenizer, model, data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path

    def _bleu4(self, pred, label):
        pred = pred.strip()
        label = label.strip()

        hypothesis = list(pred)
        reference = list(label)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0

        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        return bleu_score

    def _slot_accuracy(self, pred, label):
        correct = 0
        if pred is not None:
            for k, v in pred.items():
                if v is None:
                    continue
                if label and k in label:
                    if not isinstance(v,list):
                        correct += int(v==label[k])
                    else:
                        for t in v:
                            correct += int(t in label[k])

        pred_slots = sum(len(v) if isinstance(v, list) else 1 for v in pred.values()) if pred else 0
        true_slots = sum(len(v) if isinstance(v, list) else 1 for v in label.values()) if label else 0

        return correct, pred_slots, true_slots

    def evaluate(self):
        score_dict = {
            "slot_P": None,
            "slot_R": None,
            "slot_F1": None,
            "bleu-4": None,
        }
        bleu_scores = []
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        # 读取测试集
        with open(self.data_path,'r') as f:
            test_data = [json.loads(line) for line in f]

        import random
        random.shuffle(test_data)
        for data in tqdm(test_data):
            messages = data['messages']
            for i, message in enumerate(messages):
                if message['role'] == 'user':
                    role, end, content = get_completion(tokenizer, model, messages[:i+1])
                    content = content.strip()
                    if role != '<|assistant|>':
                        print('response is not <|assistant|> !!')
                        break
                    if end == '<|observation|>' or content.startswith('search_hotels'):
                        try:
                            _, arguments = content.split('\n')
                        except:
                            _, arguments = '','{}'
                        arguments = json.loads(arguments)
                        truth = messages[i+1]['content']
                        try:
                            _, _arguments = truth.split('\n')
                        except:
                            _, _arguments = '','{}'
                        _arguments = json.loads(_arguments)
                        correct, pred_slots, true_slots = self._slot_accuracy(arguments, _arguments)
                        true_slot_count += true_slots
                        pred_slot_count += pred_slots
                        correct_slot_count += correct
                        # reply with observation
                        if i+3 <= len(messages):
                            role, end, content = get_completion(tokenizer, model, messages[:i+3])
                            content = content.strip()
                            truth = messages[i+3]['content']
                            bleu_scores.append(self._bleu4(content, truth))
                    elif end == '<|user|>' or end == '<|endoftext|>':
                        truth = messages[i+1]['content']
                        bleu_scores.append(self._bleu4(content, truth))
        
        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = sum(bleu_scores)/len(bleu_scores)
        for k, v in score_dict.items():
            score_dict[k] = round(v * 100, 4)
        print(f"score dict: {score_dict}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, required=True, help="main model weights")
    parser.add_argument("--ckpt", type=str, default=None, required=True, help="The checkpoint path")
    parser.add_argument("--data", type=str, default=None, required=True, help="The dataset file path")
    args = parser.parse_args()

    tokenizer, model = load_model(args.model, args.ckpt)

    evaluator = Evaluator(tokenizer, model, args.data)
    evaluator.evaluate()
