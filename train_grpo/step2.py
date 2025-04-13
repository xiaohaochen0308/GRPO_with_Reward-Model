from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer
from datasets import Dataset
import torch
import json
import re
import os

# ✅ 导入你的 CTR Scorer 模块
from ctr_scorer import CTRScorer

# ✅ 准备日志文件路径
debug_log_path = "grpo_ctr_debug_log.jsonl"
# 若存在旧文件，先清空
if os.path.exists(debug_log_path):
    os.remove(debug_log_path)

# ✅ 加载你 SFT 后的全量模型（非 LoRA）
model_path = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct"        # checkpoint-4750

print("加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# ✅ 明确的 System Prompt
SYSTEM_PROMPT = """请使用以下格式用中文作答：

<think>
请写出你的思考过程，比如分析问题的背景、拆解逻辑、列出步骤等。
</think>
接着写出你最终的答案。注意：<think> 和 </think> 是必须要出现的标签，不能省略。
"""

# ✅ 加载 JSON 格式数据集
def load_custom_dataset(file_path="alpaca_train_787_sampled_full_cot.json") -> Dataset:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    processed_data = []
    for item in data:
        instruction = item["instruction"]
        output = item["output"]
        prompt = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction}
        ]
        processed_data.append({"prompt": prompt, "answer": output})

    return Dataset.from_list(processed_data)

dataset = load_custom_dataset()

# ✅ 奖励函数 1：严格格式匹配
def strict_format_reward_func(completions, **kwargs):
    pattern = r"^<think>\n.*?\n</think>\n.*"
    responses = [completion[0]["content"] for completion in completions]
    return [0.5 if re.match(pattern, r, re.DOTALL) else 0.0 for r in responses]

# ✅ 奖励函数 2：软匹配，只要有 <think> 标签并有后续答案文本
def soft_format_reward_func(completions, **kwargs):
    responses = [completion[0]["content"] for completion in completions]
    results = []
    for r in responses:
        if "<think>" in r and "</think>" in r:
            content_after = r.split("</think>")[-1].strip()
            if len(content_after) > 5:
                results.append(0.5)
            else:
                results.append(0.2)
        else:
            results.append(0.0)
    return results

# ✅ 奖励函数 3：标签计数打分
def count_xml(text) -> float:
    count = 0.0
    if text.count("<think>\n") == 1: count += 0.2
    if text.count("\n</think>\n") == 1: count += 0.2
    if len(text.split("</think>")[-1].strip()) > 0:
        count += 0.1
    return count

def xmlcount_reward_func(completions, **kwargs):
    return [count_xml(c[0]["content"]) for c in completions]

# ✅ 奖励函数 4：推理长度打分（升级阈值版）
def reasoning_length_reward_func(completions, **kwargs):
    rewards = []
    for completion in completions:
        text = completion[0]["content"]
        match = re.search(r"<think>\s*(.*?)\s*</think>", text, re.DOTALL)
        if match:
            reasoning = match.group(1).strip()
        elif "</think>" in text:
            reasoning = text.split("</think>")[0].strip()
        else:
            reasoning = ""

        length = len(reasoning)

        if length <= 150:
            reward = 0.0
        elif length <= 200:
            reward = 0.2
        elif length <= 250:
            reward = 0.4
        elif length <= 300:
            reward = 0.6
        elif length <= 350:
            reward = 0.8
        else:
            reward = 1.0

        rewards.append(reward)
    return rewards

# ✅ 奖励函数 5：CTR reward model 打分（及时写入 JSONL）
ctr_scorer = CTRScorer(
    model_path="/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct",
    ckpt_path="/root/paddlejob/workspace/env_run/LLM_seq_cls/best_model.pth",
    device="cuda"
)

def ctr_reward_func(completions, **kwargs):
    rewards = []
    prompts = kwargs.get("prompts", [""] * len(completions))

    for i, completion in enumerate(completions):
        text = completion[0]["content"]
        if "</think>" in text:
            answer = text.split("</think>")[-1].strip()
        else:
            answer = text.strip()

        score = ctr_scorer.predict(answer)
        rewards.append(score)

        # ✅ 每条即时写入 JSONL 文件
        log_item = {
            "prompt": prompts[i],
            "response": text,
            "answer": answer,
            "ctr_score": score
        }
        with open(debug_log_path, "a", encoding="utf-8") as f:
            json.dump(log_item, f, ensure_ascii=False)
            f.write("\n")

    return rewards

# ✅ GRPO 配置
training_args = GRPOConfig(
    use_vllm=True,
    learning_rate=5e-6,
    adam_beta1=0.9,
    adam_beta2=0.99,
    weight_decay=0.1,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",
    optim="adamw_torch_fused",
    logging_steps=1,
    bf16=torch.cuda.is_bf16_supported(),
    fp16=not torch.cuda.is_bf16_supported(),
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,
    num_generations=4,
    max_prompt_length=256,
    max_completion_length=200,
    max_steps=500,
    save_steps=100,
    max_grad_norm=0.1,
    report_to="none",
    output_dir="outputs",
)

# ✅ 启动 GRPO 训练
trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        reasoning_length_reward_func,
        ctr_reward_func
    ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

# ✅ 保存最终模型
model.save_pretrained("grpo_saved_model")
tokenizer.save_pretrained("grpo_saved_model")
print("✅ 模型已保存到 grpo_saved_model/")
print(f"✅ 训练过程中生成内容已记录至 {debug_log_path}")