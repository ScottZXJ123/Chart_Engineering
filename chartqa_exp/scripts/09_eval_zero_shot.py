import json, re, torch
from datasets import load_dataset, Features, Value, Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

# 1) 加载模型和 tokenizer
checkpoint = "Salesforce/blip2-flan-t5-xl"  # 或你后续用的 ChartMoE-Tiny
processor = Blip2Processor.from_pretrained(checkpoint)
model = Blip2ForConditionalGeneration.from_pretrained(checkpoint).eval().cuda()

# 2) 加载测试集
ds = load_dataset("json", data_files={"test":"data/test.json"},
                  features=Features({"image": Image(), "question": Value("string"), "answer": Value("string")}))
ds = ds["test"]

# 3) 推理与答案提取
def extract_answer(text):
    m = re.search(r"Answer\s*[:：]\s*([^\n]+)", text)
    return m.group(1).strip() if m else ""

results = []
for item in ds:
    # 构造输入
    inputs = processor(
        images=item["image"],
        text=f"Question: {item['question']}\n\nPlease write a Python 3 program to get the answer and give the value on a separate line at the end with 'Answer:'",
        return_tensors="pt"
    ).to("cuda")
    # 生成
    out = model.generate(**inputs, max_new_tokens=256)
    gen = processor.decode(out[0], skip_special_tokens=True)
    pred = extract_answer(gen)
    results.append((pred, item["answer"]))

# 4) 计算 EM
em = sum(1 for p, g in results if p == g) / len(results)
print(f"Zero-Shot PoT EM: {em:.4%}")