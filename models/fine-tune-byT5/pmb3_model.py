
import os
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, T5ForConditionalGeneration, AdamW

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


Config = {"batch_size": 32,
          "epoch_number": 10,
          "cuda_index": 0,
          "lr": 1e-4
          }


class Dataset(torch.utils.data.Dataset):
    def __init__(self, drs_file_path, text_file_path):
       
        #reading logical representation from DRS file with \n\n delimitor
        print("Reading DRS lines ...")
        with open(drs_file_path, encoding="utf-8") as f_drs:
            self.drs = f_drs.read().split("\n\n")
            

        # reading text from text file with \n delimiter
        print("Reading Text lines...")
        with open(text_file_path, encoding="utf-8") as f_text:
            self.text = f_text.read().split("\n")

        # reading logical representation from DRS file with \n\n delimiter
       # print("Reading DRS lines...")
        #with open(drs_file_path, encoding="utf-8") as f_drs:
         #   self.drs = f_drs.read().split("\n\n")

    def __len__(self):
        return min(len(self.drs),len(self.text))

    def __getitem__(self, idx):
        drs = self.drs[idx]
        text = self.text[idx]
        #drs = self.drs[idx]
        return drs, text


def get_dataloader(drs_file_path, text_file_path, batch_size=15):
    dataset = Dataset(drs_file_path, text_file_path)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    return dataloader


class Generator:

    def __init__(self, lang):
        """
        :param train: train or test
        """
        self.epoch_number = Config["epoch_number"]
        self.device = torch.device(f"cuda:{Config['cuda_index']}" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained('google/byt5-base', max_length=256)

        self.model = T5ForConditionalGeneration.from_pretrained('google/byt5-base', max_length=256)
        self.model.to(self.device)

    def evaluate(self, val_loader, save_path):
        with open(save_path, 'w+', encoding="utf-8") as f:
            self.model.eval()
            with torch.no_grad():
                for i, (drs, text) in enumerate(tqdm(val_loader)):
                    x = self.tokenizer(drs, return_tensors='pt', padding=True, truncation=True, max_length=256)['input_ids'].to(
                        self.device)
                    out_put = self.model.generate(x)
                    for j in range(len(out_put)):
                        o = out_put[j]
                        pred_text = self.tokenizer.decode(o, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                        f.write(pred_text)
                        f.write('\n')

    def train(self, train_loader, val_loader, lr, epoch_number):
        optimizer = AdamW(self.model.parameters(), lr)
        for epoch in range(epoch_number):
            self.model.train()
            pbar = tqdm(train_loader)
            for batch, (drs, text) in enumerate(pbar):
                x = self.tokenizer(drs, return_tensors='pt', padding=True, truncation=True, max_length=256)['input_ids'].to(
                    self.device)
                y = self.tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=256)['input_ids'].to(
                    self.device)

                optimizer.zero_grad()
                output = self.model(x, labels=y)
                loss = output.loss
                loss.backward()
                optimizer.step()
                pbar.set_description(f"Loss: {format(loss.item(), '.3f')}")
