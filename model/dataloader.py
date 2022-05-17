import torch
from kobert_transformers import get_tokenizer
from torch.utils.data import Dataset


class WellnessTextClassificationDataset(Dataset):
    def __init__(self,
                 file_path="./data/wellness_dialog_for_text_classification_all.txt",
                 num_label=359,
                 device='cpu',
                 max_seq_len=512,  # KoBERT max_length
                 tokenizer=None
                 ):
        self.file_path = file_path
        self.device = device
        self.data = []
        self.tokenizer = tokenizer if tokenizer is not None else get_tokenizer()

        file = open(self.file_path, 'r', encoding='utf-8')

        while True:
            line = file.readline()
            if not line:
                break
            datas = line.split("    ")
            index_of_words = self.tokenizer.encode(datas[0])
            token_type_ids = [0] * len(index_of_words)
            attention_mask = [1] * len(index_of_words)

            # Padding Length
            padding_length = max_seq_len - len(index_of_words)

            # Zero Padding
            index_of_words += [0] * padding_length
            token_type_ids += [0] * padding_length
            attention_mask += [0] * padding_length

            # Label
            label = int(datas[1][:-1])

            data = {
                'input_ids': torch.tensor(index_of_words).to(self.device),
                'token_type_ids': torch.tensor(token_type_ids).to(self.device),
                'attention_mask': torch.tensor(attention_mask).to(self.device),
                'labels': torch.tensor(label).to(self.device)
            }

            self.data.append(data)

        file.close()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        return item


if __name__ == "__main__":
    dataset = WellnessTextClassificationDataset()
    print(dataset)
