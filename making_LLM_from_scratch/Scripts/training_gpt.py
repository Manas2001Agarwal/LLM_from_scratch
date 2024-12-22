import torch
import math
import tiktoken
import torch.nn as nn
from making_LLM_from_scratch.Scripts.text_preprocessing import create_dataloader
from making_LLM_from_scratch.Scripts.gpt_archietecture import GPTModel
torch.manual_seed(123)
tokenizer = tiktoken.get_encoding("gpt2")

cfg = {
    "vocab_size": 50257,    # Vocabulary size
    "context_length": 256, # Context length
    "emb_dim": 768,         # Embedding dimension
    "n_heads": 12,          # Number of attention heads
    "n_layers": 12,         # Number of layers
    "drop_rate": 0.1,       # Dropout rate
    "qkv_bias": False       # Query-Key-Value bias
}

def generate_simple_text(model,idx,max_tokens,content_size):
    for _ in range(max_tokens):
        idx = idx[:,-content_size:]
        logits = model(idx)[:,-1,:]
        probas = torch.softmax(logits,dim=-1)
        token_id = torch.argmax(probas,dim=-1,keepdim=True)
        idx = torch.cat((idx,token_id),dim=1)
    return idx

def text_to_token_ids(text,tokenizer):
    tokens = tokenizer.encode(text,allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(tokens).unsqueeze(0)   ## Adding Batch Dimension
    return encoded_tensor

def token_ids_to_text(token_ids,tokenizer):
    token_ids_flat = token_ids.squeeze(0)
    text = tokenizer.decode(token_ids_flat.tolist())
    return text

def calc_batch_loss(input_batch,target_batch,model,device):
    input_batch = input_batch.to(device=device)
    target_batch = target_batch.to(device=device)
    
    logits = model(input_batch)
        
    loss = torch.nn.functional.cross_entropy(logits.flatten(start_dim=0,end_dim=1),
                                             target_batch.flatten())
    return loss

def calc_loss_loader(data_loader,model,device,num_batches=None):
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches,len(data_loader))
        
    loss_batch = []
        
    for i, (inputs_id,target_id) in enumerate(data_loader):
        if i < num_batches:
            logits = model(inputs_id)
            
            loss = torch.nn.functional.cross_entropy(logits.flatten(start_dim=0,end_dim=1),
                                             target_id.flatten())
            loss_batch.append(loss)
        else:
            break
        
    return torch.mean(torch.tensor(loss_batch)).item()        

def evaluate_model(model,train_loader,val_loader,device,eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader,model,device)
        val_loss = calc_loss_loader(val_loader,model,device)
        
    model.train()
    return train_loss,val_loss

def generate_and_print_simple_text(start_context,tokenizer,model):
    model.eval()
    idx = torch.tensor(tokenizer.encode(start_context)).unsqueeze(0)
    content_size = 256
    output_id = generate_simple_text(model,idx,max_tokens=50,
                                     content_size = content_size)
    text = token_ids_to_text(output_id,tokenizer)
    print(text.replace("\n"," "))   
    model.train()
    
def training_gpt(model,train_dl,val_dl,optimizer,device,
                 eval_freq,eval_iter,num_epochs,
                 start_context = "Every effort moves", tokenizer=tokenizer):
    track_train_loss,track_val_loss,track_token_count = [],[],[]
    tokens_seen,global_step_id = 0, -1
    
    model.train()
    for epoch in range(num_epochs):
        for input_ids,target_ids in train_dl:
            optimizer.zero_grad()
            
            loss = calc_batch_loss(input_ids,target_ids,model,device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_ids.numel()
            global_step_id += 1
            
            if global_step_id%eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_dl,val_dl,device,eval_iter)
                track_train_loss.append(train_loss)
                track_val_loss.append(track_val_loss)
                track_token_count.append(tokens_seen)
                
                print(f"Ep {epoch+1} (Step {global_step_id:06d}): "
                 f"Train loss {train_loss:.3f}, "
                 f"Val loss {val_loss:.3f}")
                
        generate_and_print_simple_text(start_context,tokenizer,model)
    return track_train_loss,track_val_loss,track_token_count

def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):

    # For-loop is the same as before: Get logits, and only focus on last time step
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]

        # New: Filter logits with top_k sampling
        if top_k is not None:
            # Keep only top_k values
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float("-inf")).to(logits.device), logits)

        # New: Apply temperature scaling
        if temperature > 0.0:
            logits = logits / temperature

            # Apply softmax to get probabilities
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # Otherwise same as before: get idx of the vocab entry with the highest logits value
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        if idx_next == eos_id:  # Stop generating early if end-of-sequence token is encountered and eos_id is specified
            break

        # Same as before: append sampled index to the running sequence
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx

def modified_training_gpt(model,train_dl,val_dl,optimizer,device,
                 eval_freq,eval_iter,num_epochs,
                 start_context = "Every effort moves", tokenizer=tokenizer,
                 initial_lr=3e-5,min_lr=1e-6,warmup_step = 20):
    track_train_loss,track_val_loss,track_token_count = [],[],[]
    tokens_seen,global_step_id = 0, -1
    
    track_lr = []
    total_training_steps = len(train_dl)*num_epochs
    peak_lr = optimizer.param_groups[0]["lr"]
    lr_increment = (peak_lr - initial_lr)/warmup_step
    
    model.train()
    for epoch in range(num_epochs):
        for input_ids,target_ids in train_dl:
            optimizer.zero_grad()
            global_step_id += 1
            
            if global_step_id > warmup_step:
                lr = initial_lr + global_step_id*lr_increment
                
            else:
                progress = ((global_step_id - warmup_step)/(total_training_steps-warmup_step))
                lr = min_lr + (peak_lr-min_lr)*0.5*(1+math.cos(math.pi*progress))
                
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            
            loss = calc_batch_loss(input_ids,target_ids,model,device)
            loss.backward()
            
            if global_step_id > warmup_step:
                torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1)
            
            optimizer.step()
            tokens_seen += input_ids.numel()
            
            
            if global_step_id%eval_freq == 0:
                train_loss,val_loss = evaluate_model(model,train_dl,val_dl,device,eval_iter)
                track_train_loss.append(train_loss)
                track_val_loss.append(track_val_loss)
                track_token_count.append(tokens_seen)
                
                print(f"Ep {epoch+1} (Step {global_step_id:06d}): "
                 f"Train loss {train_loss:.3f}, "
                 f"Val loss {val_loss:.3f}")
                
        generate_and_print_simple_text(start_context,tokenizer,model)
    return track_train_loss,track_val_loss,track_token_count,track_lr

if __name__ == "__main__":
    with open("/Users/mukulagarwal/Desktop/Projects/transformers_/making_LLM_from_scratch/the-verdict.txt") as file:
        text = file.read()
    
    train_ratio = int(len(text)*0.90)
    train_data = text[:train_ratio]
    val_data = text[train_ratio:]
    
    train_dl = create_dataloader(
        text = train_data,
        batch_size=2,
        maxlength=cfg['context_length'],
        stride=cfg['context_length'],
        shuffle=True,
        num_workers=0
    )

    val_dl = create_dataloader(
        text = val_data,
        batch_size=2,
        maxlength=cfg['context_length'],
        stride=cfg['context_length'],
        shuffle=False,
        num_workers=0
    )
    
    torch.manual_seed(123)
    device = torch.device("cpu")
    
    model = GPTModel(cfg)
    model.to(device)
    device = torch.device("cpu")
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004, weight_decay=0.1
    )
    
    train_losses, val_losses, tokens_seen= training_gpt(model, train_dl, val_dl, optimizer, device,
                                                        eval_freq=5, eval_iter=5,num_epochs=10,
                                                        start_context="Every effort moves you", tokenizer=tokenizer )
    print(f"Train Losses: {train_losses}")
    print(f"Validation Losses: {val_losses}")
    
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict" : optimizer.state_dict()
        },
        "model_and_optimizer.pth"
    )
    
    