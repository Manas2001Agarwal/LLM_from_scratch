# LLM_from_scratch

In this project I created a Decoder based GPT LLM right from scratch using PyTorch. **Utilized GPT2 BPE tokenizer** to preprocess text data and successfully Pretrained my GPT model on short story "verdict.txt" to perform next word prediction. Implemeted the scaled dot product multihead attention (MHA) in from scratch, adding a causal mask to prevent LLM from accessing future tokens. 

Assembled and placed all components of LLM (including LayerNorm and FeedForward Module along with MHA) in correct order plus adding Dropout and skip connections according to archietecture. Pretrained the model using Cross-Entropy loss over the next-word predicted as the objective function. **Utilized learning-rate warmup, cosine daecay and gradient clipping techniques for efficient model training**.

Adding [Probabilistic + top-K] sampling and temperatue scaling to diversify the generated text while maintaining context during inference.

# Classification Fine-Tuning
For this exercise I loaded the publically available weights of GPT-2 small(124M) model, replaced the last layer with a classification head and fine tuned the model on SMS SPAM classification dataset. I got an accuracy of 84.56% on test data.

Futher utilized advanced **PEFT based LORA technique** to fine-tune the GPT-2 small(124M) model to acheive an accuracy of 97.5% on test dataset. As a last experiment I utilized the **Distill-Bert pretrained model from HuggingFace** and fine-tuned it on SMS Spam Classification to acheive an accuracy of 98.5% on test data. 

Once again we see encoder based Bert Model having won the race for representational tasks like Text-Classification.  
Code for Classification Fine Tuning can be found in FineTing folder of this repo
