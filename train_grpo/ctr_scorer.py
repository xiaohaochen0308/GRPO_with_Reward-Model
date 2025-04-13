# ctr_scorer.py
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import re

class ModelWithLoraRegressor(nn.Module):
    def __init__(self, base_model):
        super(ModelWithLoraRegressor, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1, dtype=torch.bfloat16)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        avg_hidden_state = torch.mean(hidden_states, dim=1)
        logits = self.mlp(avg_hidden_state)
        return logits

class CTRScorer:
    def __init__(self, model_path, ckpt_path, device="cuda"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            output_hidden_states=True
        )

        lora_config = LoraConfig(
            peft_type="LORA",
            task_type="SEQ_CLS",
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
        )

        lora_model = get_peft_model(base_model, lora_config)
        self.model = ModelWithLoraRegressor(lora_model)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=device))
        self.model.to(device)
        self.model.eval()
        self.device = device

    def predict(self, text):
        inputs = self.tokenizer(text, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            prob = torch.sigmoid(logits).squeeze().item()
        return float(prob)