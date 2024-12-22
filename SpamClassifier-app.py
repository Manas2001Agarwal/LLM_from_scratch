from pathlib import Path
import sys

import tiktoken     
import torch       
import chainlit     
from making_LLM_from_scratch.Scripts.gpt_archietecture import GPTModel

def classify_review(text,model,tokenizer,max_length,pad_token_id = 50256):
    
    input_tokens = tokenizer.encode(text)
    sup_context_length = model.pos_emb.weight.shape[0]
    input_ids = input_tokens[:min(max_length,sup_context_length)]
    
    input_ids += [pad_token_id]*(max_length-len(input_ids))
    input_ids = torch.tensor(input_ids,device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_ids)[:,-1,:]
        
    predicted_labels = torch.argmax(logits,dim=-1,keepdim=True).item()
    return "spam" if predicted_labels == 1 else "not spam"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load finetuned GPT-2 model generated in chapter 6.
    This requires that you run the code in chapter 6 first, which generates the necessary model.pth file.
    """

    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Context length
        "emb_dim": 768,          # Embedding dimension
        "n_heads": 12,           # Number of attention heads
        "n_layers": 12,          # Number of layers
        "drop_rate": 0.1,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    tokenizer = tiktoken.get_encoding("gpt2")

    #model_path = Path("..") / "FineTuning" / "review_classifier.pth"
    model_path = Path("/Users/mukulagarwal/Desktop/Projects/transformers_/FineTuning/review_classifier.pth")
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file."
        )
        sys.exit()

    # Instantiate model
    model = GPTModel(GPT_CONFIG_124M)

    
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # Then load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return tokenizer, model


# Obtain the necessary tokenizer and model files for the chainlit function below
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    user_input = message.content

    label = classify_review(user_input, model, tokenizer, max_length=120)

    await chainlit.Message(
        content=f"{label}",  # This returns the model response to the interface
    ).send()