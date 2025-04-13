from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.optim import AdamW
from peft import get_peft_model, LoraConfig
import torch.nn as nn
import torch
from torch.utils.data import DataLoader, Subset
from data import train_dataloader, test_dataloader
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score


class ModelWithLoraClassifier(nn.Module):
    def __init__(self, base_model, num_classes):
        super(ModelWithLoraClassifier, self).__init__()
        self.base_model = base_model
        hidden_size = base_model.config.hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, 1024, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512, dtype=torch.bfloat16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes, dtype=torch.bfloat16)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        avg_hidden_state = torch.mean(hidden_states, dim=1)
        logits = self.mlp(avg_hidden_state)
        return logits.view(-1)  # [B]


def evaluate(model, dataloader, criterion, device, desc="Evaluating"):
    model.eval()
    eval_loss = 0
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc=desc):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)

            eval_loss += loss.item()
            probs = torch.sigmoid(logits)

            all_probs.extend(probs.float().cpu().numpy())       # ✅ bfloat16 → float
            all_labels.extend(labels.float().cpu().numpy())     # ✅ 保证 float

    avg_loss = eval_loss / len(dataloader)
    preds = (torch.tensor(all_probs) > 0.5).long()
    labels_tensor = torch.tensor(all_labels).long()
    accuracy = (preds == labels_tensor).float().mean().item()

    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, accuracy, auc


def train_and_evaluate(model, train_loader, test_loader, optimizer, criterion, device, num_epochs, writer,
                       save_path="best_model.pth", val_loader=None, val_every_n_steps=200):
    best_accuracy = 0
    global_step = 0

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []

        print(f"\nEpoch {epoch+1} - Total parameters: {sum(p.numel() for p in model.parameters())}, "
              f"Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            global_step += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device).float()

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.detach().float().cpu().numpy())   # ✅ float()
            all_labels.extend(labels.detach().float().cpu().numpy()) # ✅ float()

            if global_step % 10 == 0:
                accuracy = (torch.tensor(all_preds) == torch.tensor(all_labels)).float().mean().item()
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Step {global_step}: Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
                writer.add_scalar('Train/Batch_Loss', avg_loss, global_step)
                writer.add_scalar('Train/Batch_Accuracy', accuracy, global_step)

            if epoch > 0 and val_loader and global_step % val_every_n_steps == 0:
                val_loss, val_acc, val_auc = evaluate(model, val_loader, criterion, device, desc="Validating")
                print(f"Step {global_step} - Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}")
                writer.add_scalar('Val/Loss', val_loss, global_step)
                writer.add_scalar('Val/Accuracy', val_acc, global_step)
                writer.add_scalar('Val/AUC', val_auc, global_step)

        test_loss, test_accuracy, test_auc = evaluate(model, test_loader, criterion, device, desc="Testing")
        print(f"Epoch {epoch+1} - Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test AUC: {test_auc:.4f}")
        writer.add_scalar('Test/Loss', test_loss, epoch)
        writer.add_scalar('Test/Accuracy', test_accuracy, epoch)
        writer.add_scalar('Test/AUC', test_auc, epoch)

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(model.state_dict(), save_path)
            print("Best model saved.")

    writer.close()
    return best_accuracy


if __name__ == "__main__":
    writer = SummaryWriter('logs/')

    model_name = "/root/.cache/modelscope/hub/models/Qwen/Qwen2.5-3B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        output_hidden_states=True
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    lora_config = LoraConfig(
        peft_type="LORA",
        task_type="SEQ_CLS",
        r=16,
        lora_alpha=32,
        lora_dropout=0.05
    )
    lora_model = get_peft_model(model, lora_config)

    num_classes = 1
    model_with_classifier = ModelWithLoraClassifier(lora_model, num_classes)

    optimizer = AdamW(model_with_classifier.parameters(), lr=5e-5)
    criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_with_classifier.to(device)

    # ✅ Debug模式：仅用前100条样本
    # train_dataset = Subset(train_dataloader.dataset, range(100))
    # test_dataset = Subset(test_dataloader.dataset, range(100))
    # train_dataloader = DataLoader(train_dataset, batch_size=train_dataloader.batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=test_dataloader.batch_size, shuffle=False)
    val_dataset = Subset(test_dataloader.dataset, range(min(100, len(test_dataloader.dataset))))
    val_dataloader = DataLoader(val_dataset, batch_size=test_dataloader.batch_size, shuffle=False)

    num_epochs = 3
    best_accuracy = train_and_evaluate(
        model_with_classifier,
        train_dataloader,
        test_dataloader,
        optimizer,
        criterion,
        device,
        num_epochs,
        writer,
        save_path="best_model.pth",
        val_loader=val_dataloader,
        val_every_n_steps=200
    )

    print(f"Best Test Accuracy: {best_accuracy:.4f}")