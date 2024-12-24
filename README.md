# LLM_from_scratch

In this project I created an Decoder based GPT LLM right from scratch using PyTorch. Utilized GPT2 BPE tokenizer to preprocess text data and successfully Pretrained my GPT model on short story "verdict.txt" to perform next word prediction. Implemeted the scaled dot product multihead attention (MHA) in from scratch, adding a causal mask to prevent LLM from accessing future tokens. 

Assembled and placed all components of LLM (including LayerNorm and FeedForward Module along with MHA) in correct order plus adding Dropout and skip connections according to archietecture. Pretrained the model using Cross-Entropy loss over the next-word predicted as the objective function. Utilized learning-rate warmup, cosine daecay and gradient clipping techniques for efficient model training.

Adding [Probabilistic + top-K] sampling and temperatue scaling to diversify the generated text while maintaining context during inference.
