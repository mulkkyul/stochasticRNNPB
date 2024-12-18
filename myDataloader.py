import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import glob
import os
import pandas as pd
from typing import Tuple, List
from utils import load_params

# It is better to use typing for complex structures like the data input.
from typing import Sequence

# Custom Dataset class
class MinimalDataset(Dataset):
    def __init__(self, data_input: Sequence[torch.Tensor], data_labels: Sequence[int], device='cuda'):
        self.data_input = data_input
        self.data_label = data_labels
        self.device = device


    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        return self.data_input[index].to(self.device), self.data_label[index]

    def __len__(self) -> int:
        return len(self.data_input)

# Custom collate function
def pad_collate(batch: List[Tuple[torch.Tensor, int]]) -> Tuple[torch.Tensor, torch.Tensor, List[int]]:
    xx, yy = zip(*batch)
    xx_pad = pad_sequence(xx, batch_first=False, padding_value=0)
    yy = torch.tensor(yy, dtype=torch.long)  # Convert labels to tensor
    x_lens = torch.tensor([len(x) for x in xx], dtype=torch.long)  # Convert sequence lengths to tensor
    return xx_pad, yy, x_lens

def load_dataset(path_dataset, device):

    print(f"Loading the data from {path_dataset}")
    files_dataset = sorted(glob.glob(os.path.join(path_dataset, "*.csv")))
    num_files = len(files_dataset)
    if num_files == 0:
        print("No datasets found.")
        raise FileNotFoundError("No datasets found matching the pattern.")

    data_input = []
    data_label = []
    data_filename = []
    for idx_file, csv_filename in enumerate(files_dataset):
        df = pd.read_csv(csv_filename, header=None)
        data_input.append(torch.tensor(df.values, dtype=torch.float32))
        data_label.append(idx_file)
        data_filename.append(csv_filename)

    dataset = MinimalDataset(data_input, data_label, device)
    data_dim = df.shape[1]
    print(f"Loaded {len(dataset)} sequences.")
    print("=" * 80)
    return dataset, data_filename

if __name__ == "__main__":
    params = load_params("./params.json")

    path_dataset = params["path_dataset_train"]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset, data_filename, data_dim = load_dataset(path_dataset, device)
    print(data_filename)

    data_loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=pad_collate)
    for i, (x_padded, y, x_lens) in enumerate(data_loader):
        print(f'Batch {i}:')
        print(f'Input sequences (padded): {x_padded}')
        print(f'Labels: {y}')
        print(f'Sequence lengths: {x_lens}')
        print(len(dataset))
