import time
import os
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import get_cosine_schedule_with_warmup
from .device import get_device
import math

class Trainer:
    learning_rate = 1e-4
    weight_decay = 1e-2
    betas = (0.9, 0.999)
    grad_clip = 1.0
    max_epochs = 20
    patience = 5
    save_interval_multiplier = 0.01

    def __init__(self, model, splits, tokenizer, checkpoint=None):
        self.model = model
        self.train_loader = splits["train"]
        self.val_loader = splits["val"]
        self.tokenizer = tokenizer
        self.device = get_device()
        self.model.to(self.device)

        self.num_training_steps = len(self.train_loader)
        self.num_save_steps = int(self.num_training_steps * self.save_interval_multiplier)
        self.early_stopping_counter = 0

        self.num_warmup_steps = int(self.num_training_steps * 0.1)

        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=self.betas
        )
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=self.num_training_steps
        )

        self.save_path = f"checkpoints/{self.model.config.name}.pt"
        self.resume_checkpoint = checkpoint is not None
        if self.resume_checkpoint:
            self.checkpoint = checkpoint
        else:
            self.checkpoint = {
                'epoch': 0,
                'step': 0,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'history': {
                    'train_loss': [],
                    'train_perplexity': [],
                    'val_loss': [],
                    'val_perplexity': [],
                }
            }
    
    
    def _save_checkpoint(self):
        if not os.path.exists(self.save_path):
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        torch.save(self.checkpoint, self.save_path)

    def _get_time_remaining(self, i, start_time):
        time_per_step = (time.time() - start_time) / (i + 1)
        time_remaining = time_per_step * (len(self.train_loader) - i)
        hours = int(time_remaining / 3600)
        minutes = int((time_remaining % 3600) / 60)
        seconds = int(time_remaining % 60)
        return f"{hours}:{minutes}:{seconds}"

    def _step(self, batch):
        input_ids = batch.to(self.device)
        x = input_ids[:, :-1]
        targets = input_ids[:, 1:]
        _, loss = self.model(x, targets=targets, ignore_index=self.tokenizer.pad_token_id)
        return loss

    def validate(self):
        self.model.eval()
        loss = 0.0
        with torch.no_grad():
            for batch in self.val_loader:
                loss += self._step(batch).item()
        self.model.train()
        return loss / len(self.val_loader)

    def train(self):
        self.model.train()
        
        start_time = time.time()
        
        val_loss = float("inf")
        val_perplexity = float("inf")
        best_val_loss = float("inf")

        for epoch in range(self.max_epochs):
            
            if self.resume_checkpoint and epoch <= self.checkpoint["epoch"]:
                print(f"Skipping epoch {epoch} (already completed)")
                continue
            
            for i, batch in enumerate(self.train_loader):
                
                if self.resume_checkpoint and epoch == self.checkpoint["epoch"] and i <= self.checkpoint["step"]:
                    print(f"\r[Epoch {epoch} | Step {i}/{len(self.train_loader)}] Skipping (already completed)", end="")
                    continue

                train_loss = self._step(batch)
        
                self.optimizer.zero_grad()
                train_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.optimizer.step()
                self.scheduler.step()
                train_loss = train_loss.item()

                if i != 0 and (i % self.num_save_steps == 0 or i == len(self.train_loader) - 1):
                    val_loss = self.validate()
                    val_perplexity = math.exp(min(val_loss, 100))
                    step = i + epoch * len(self.train_loader)
                    self.checkpoint["history"]["train_loss"].append((step, train_loss))
                    self.checkpoint["history"]["val_loss"].append((step, val_loss))
                    self.checkpoint["history"]["val_perplexity"].append((step, val_perplexity))
                    self.checkpoint["epoch"] = epoch
                    self.checkpoint["step"] = i
                    self.checkpoint["model_state_dict"] = self.model.state_dict()
                    self.checkpoint["optimizer_state_dict"] = self.optimizer.state_dict()
                    self.checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()
                    
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        self.early_stopping_counter = 0
                        self._save_checkpoint()
                    else:
                        self.early_stopping_counter += 1
                    if self.early_stopping_counter >= self.patience:
                        print(f"Early stopping triggered after {epoch} epochs")
                        break
                    
                time_remaining = self._get_time_remaining(i, start_time)
                print(f"\r[Epoch {epoch} | Step {i}/{len(self.train_loader)}] train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val ppl: {val_perplexity:.4f} | time remaining: {time_remaining}", end="")
            
            print(f"Epoch {epoch} | train loss: {train_loss:.4f} | val loss: {val_loss:.4f} | val ppl: {val_perplexity:.4f} | best val loss: {best_val_loss:.4f}")