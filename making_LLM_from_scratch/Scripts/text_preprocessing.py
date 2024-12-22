import torch
from torch.utils.data import DataLoader,Dataset
import tiktoken
import urllib.request

class GPTDatasetV1(Dataset):
    def __init__(self,text,tokenizer,maxlength,stride):
        self.input_ids = []
        self.target_ids = []
        
        self.token_ids = tokenizer.encode(text)
        
        for i in range(0,len(self.token_ids)-maxlength,stride):
            input_ids = self.token_ids[i:i+maxlength]
            target_ids = self.token_ids[i+1:i+maxlength+1]
            self.input_ids.append(input_ids)
            self.target_ids.append(target_ids)
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return torch.tensor(self.input_ids[index]),torch.tensor(self.target_ids[index])
    
def create_dataloader(text,batch_size=4,maxlength=256,
                      stride=128,drop_last=True,
                      num_workers = 0,
                      shuffle = False):
    
    tokenizer = tiktoken.get_encoding("gpt2")
    
    return DataLoader(
        dataset=GPTDatasetV1(text=text,tokenizer=tokenizer,maxlength=maxlength,stride=stride),
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=drop_last,
        shuffle = shuffle
    )
    
if __name__ == "__main__":
    with open('the-verdict.txt','r',encoding = 'utf-8') as file:
        text = file.read()
    
    train_dl = create_dataloader(text,batch_size=8,maxlength=4,stride=4,shuffle=False)
    input_text,target_text = next(iter(train_dl))
    
    vocab_size = 50257
    output_dim = 256
    context_length = 4
    token_embedding_layer = torch.nn.Embedding(vocab_size,output_dim)
    position_embedding_layer = torch.nn.Embedding(context_length,output_dim)
    
    token_embeddings = token_embedding_layer(input_text)
    position_embedding = position_embedding_layer(torch.arange(context_length))
    
    print((token_embeddings+position_embedding).shape)
    
        