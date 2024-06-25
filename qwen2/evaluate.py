import json
import torch
import argparse
from tqdm import tqdm
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_preprocess import build_prompt, parse_json

def load_model(model_path, checkpoint_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, model_id=checkpoint_path).to("cuda").eval()
    return tokenizer, model

class Evaluator:
    def __init__(self,tokenizer,model,data_path):
        self.tokenizer = tokenizer
        self.model = model
        self.data_path = data_path

    def slot_accuracy(self, pred, label):
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

    def bleu4(self, pred, label):
        pred = pred.strip()
        label = label.strip()

        hypothesis = list(pred)
        reference = list(label)

        if len(hypothesis) == 0 or len(reference) == 0:
            return 0

        bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
        return bleu_score

    def compute_metrics(self):
        score_dict = { "slot_P": 0.0, "slot_R": 0.0, "slot_F1": 0.0 }
        bleu_scores = []
        true_slot_count = 0
        pred_slot_count = 0
        correct_slot_count = 0

        with open(self.data_path, "r", encoding="utf-8") as f:
            test_dataset = [json.loads(line) for line in f]

        for item in tqdm(test_dataset):
            template = build_prompt(item["context"])
            inputs = self.tokenizer([template], return_tensors="pt").to("cuda")
            with torch.no_grad():
                outputs = self.model.generate(**inputs, max_new_tokens=1024)
                response = self.tokenizer.decode(outputs[:,inputs['input_ids'].shape[1]:][0], skip_special_tokens=True)
            label = json.loads(item["response"])
            if label["role"] == "search":
                try:
                    preds = parse_json(response)
                except:
                    preds = {}
                truth = label["arguments"]
                correct, pred_slots, true_slots = self.slot_accuracy(preds, truth)
                true_slot_count += true_slots
                pred_slot_count += pred_slots
                correct_slot_count += correct
            else:
                response = response.replace("assistant","")
                bleu_scores.append(self.bleu4(response, label['content']))

        score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
        score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
        score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
        score_dict["bleu-4"] = sum(bleu_scores) / len(bleu_scores)
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
    evaluator.compute_metrics()
