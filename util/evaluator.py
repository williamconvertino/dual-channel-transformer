import random
import torch
import torch.nn.functional as F
from .trainer import Trainer
from .device import get_device

def generate_text_greedy(model, tokenizer, prompt, max_length=50, temperature=1.0):
    
    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        probabilities = F.softmax(next_token_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist()[input_size:])

def generate_text_beam(model, tokenizer, prompt, max_length=50, beam_width=3):
    
    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    input_size = input_ids.size(1)
    beams = [(input_ids, 0)] 
    
    for _ in range(max_length):
        new_beams = []
        
        for seq, score in beams:
            if seq.size(1) > model.config.max_seq_len:
                seq_input = seq[:, -model.config.max_seq_len:]
            else:
                seq_input = seq
            
            logits, _ = model(seq_input)
            next_token_logits = logits[:, -1, :]
            log_probs = F.log_softmax(next_token_logits, dim=-1)
            
            topk_log_probs, topk_indices = torch.topk(log_probs, beam_width, dim=-1)
            
            for i in range(beam_width):
                next_token = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                new_seq = torch.cat([seq, next_token], dim=1).to(device)
                new_score = score + topk_log_probs[0, i].item()
                new_beams.append((new_seq, new_score))
        
        new_beams = sorted(new_beams, key=lambda x: x[1], reverse=True)[:beam_width]
        beams = new_beams
        
        if all(seq[0, -1].item() == tokenizer.eos_token_id for seq, _ in beams):
            break

    best_seq = beams[0][0]
    return tokenizer.decode(best_seq[0].tolist()[input_size:])

def generate_text_topk(model, tokenizer, prompt, max_length=50, temperature=1.0, top_k=40):
    
    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated

        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        
        topk_values, topk_indices = torch.topk(next_token_logits, top_k, dim=-1)
        filtered_logits = torch.full_like(next_token_logits, float('-inf'))
        filtered_logits.scatter_(dim=-1, index=topk_indices, src=next_token_logits.gather(dim=-1, index=topk_indices))
        probabilities = F.softmax(filtered_logits, dim=-1)
        
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
    
    return tokenizer.decode(generated[0].tolist()[input_size:])

def generate_text_nucleus(model, tokenizer, prompt, max_length=50, temperature=1.0, top_p=0.9):

    model.eval()
    
    input_ids = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(device)
    generated = input_ids
    input_size = input_ids.size(1)
    
    for _ in range(max_length):
        if generated.size(1) > model.config.max_seq_len:
            input_ids = generated[:, -model.config.max_seq_len:]
        else:
            input_ids = generated
            
        logits, _ = model(input_ids)
        next_token_logits = logits[:, -1, :] / temperature
        
        probs = F.softmax(next_token_logits, dim=-1)
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        indices_to_remove = torch.zeros_like(next_token_logits, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
        
        probabilities = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probabilities, num_samples=1)
        generated = torch.cat((generated, next_token), dim=1).to(device)
        
        if next_token.item() == tokenizer.eos_token_id:
            break
            
    return tokenizer.decode(generated[0].tolist()[input_size:])

class Evaluator:
    def __init__(self, model, splits, tokenizer, checkpoint=None):
        self.model = model
        self.splits = splits
        self.tokenizer = tokenizer
        self.checkpoint = checkpoint

        self.device = get_device()
        
    def _get_test_loss(self):
        
        test_loss = 0.0
        
        with torch.no_grad():
            for batch in self.splits["test"]:
                input_ids = batch.to(self.device)
                x = input_ids[:, :-1]
                targets = input_ids[:, 1:]
                _, loss = self.model(x, targets=targets, ignore_index=self.tokenizer.pad_token_id)
                test_loss += loss.item()
                
        return test_loss / len(self.splits["test"])

    def evaluate(self, num_prompts=10):
        
        if self.checkpoint:
            self.model.load_state_dict(self.checkpoint["model_state_dict"])
        
        self.model.to(self.device)
        self.model.eval()

        test_loss = self._get_test_loss()
        
        print(f"=" * 50)
        print(f"Test Loss: {test_loss:.4f}")
        print(f"=" * 50)
        
        prompts = []
        
        for i, batch in enumerate(self.splits["test"]):
            if i >= num_prompts:
                break
            
            example = batch[0].tolist()
            example = [token for token in example if token != self.tokenizer.pad_token_id]
            max_tokens = min(self.model.config.max_seq_len // 2, len(example) // 2)

            example = example[:max_tokens]            
            
            prompts.append(self.tokenizer.decode(example))
            
        for prompt in prompts:
            print("=" * 50)
            print("Prompt:")
            print(prompt)
            print("-" * 50)
            
            generated_text = generate_text_nucleus(self.model, self.tokenizer, prompt, max_length=50)
            print("Generated Text:")
            print(generated_text)
            print("=" * 50)