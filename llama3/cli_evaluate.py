import json
import torch
from tqdm import tqdm
from peft import PeftModel, LoraConfig, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from data_preprocess import build_prompt

RAW_MODEL_PATH = "/root/autodl-tmp/Meta-Llama-3-8B-Instruct"
ADAPTER_PATH = "output/hotel_qlora-20240429-160008/checkpoint-2100"

def slot_accuracy(pred, label):
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

def bleu4(pred, label):
    pred = pred.strip()
    label = label.strip()

    hypothesis = list(pred)
    reference = list(label)

    if len(hypothesis) == 0 or len(reference) == 0:
        return 0

    bleu_score = sentence_bleu([reference], hypothesis, smoothing_function=SmoothingFunction().method3)
    return bleu_score

def load_model(model_name):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )
    n_gpus = torch.cuda.device_count()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto", # dispatch efficiently the model on the available ressources
        max_memory = {i: '24500MB' for i in range(n_gpus)},
    )
    # model.enable_input_require_grads()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer  = load_model(RAW_MODEL_PATH)
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM, inference_mode=False,
    target_modules=["q_proj", "k_proj", "v_proj"],
    r=8, lora_alpha=32, lora_dropout=0.1
)
model = PeftModel.from_pretrained(model, model_id=ADAPTER_PATH, config=lora_config)

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]
score_dict = { "slot_P": 0, "slot_R": 0, "slot_F1": 0 }
bleu_scores = []
true_slot_count = 0
pred_slot_count = 0
correct_slot_count = 0

with open("../data/test.llama3.jsonl", "r", encoding="utf-8") as f:
    test_dataset = [json.loads(line) for line in f]

for item in tqdm(test_dataset):
    template = build_prompt(item["context"])
    input_ids = tokenizer.encode(template, add_special_tokens=False, return_tensors='pt').cuda()
    outputs = model.generate(
        input_ids=input_ids, max_new_tokens=512, 
        eos_token_id=terminators,
        pad_token_id=tokenizer.eos_token_id
    )
    outputs = outputs.tolist()[0][len(input_ids[0]):]
    response = tokenizer.decode(outputs, skip_special_tokens=True)
    label = json.loads(item["response"])
    if label["role"] == "search":
        try:
            preds = json.loads(response.strip()[7:])
        except:
            preds = {}
        truth = label["arguments"]
        correct, pred_slots, true_slots = slot_accuracy(preds, truth)
        true_slot_count += true_slots
        pred_slot_count += pred_slots
        correct_slot_count += correct
    else:
        bleu_scores.append(bleu4(response, label['content']))

score_dict["slot_P"] = float(correct_slot_count/pred_slot_count) if pred_slot_count > 0 else 0
score_dict["slot_R"] = float(correct_slot_count/true_slot_count) if true_slot_count > 0 else 0
score_dict["slot_F1"] = 2*score_dict["slot_P"]*score_dict["slot_R"]/(score_dict["slot_P"]+score_dict["slot_R"]) if (score_dict["slot_P"]+score_dict["slot_R"]) > 0 else 0
score_dict["bleu-4"] = sum(bleu_scores) / len(bleu_scores)
for k, v in score_dict.items():
    score_dict[k] = round(v * 100, 4)
print(f"score dict: {score_dict}")
