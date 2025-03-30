from torch.utils.data import Dataset

class customDataset(Dataset):
  def __init__(self, data):
    super().__init__()
    self.data = data

  def __len__(self):
    return len(self.data)
  
  def __getitem__(self, idx):
    return self.data[idx]


from torch.utils.data import Dataset

class DataLoader:

  def __init__(self, dataset:Dataset, batch_size=16):
    self.data = dataset
    self.batch_size = batch_size

  def __iter__(self):
    for i in range(0, len(self.data), self.batch_size):
      yield self.data[i: i+self.batch_size]


from torch.utils.data import Dataset
from transformers import DefaultDataCollator

class DataLoader:

  def __init__(self, dataset:Dataset, batch_size=16, collate_fn=DefaultDataCollator):
    self.data = dataset
    self.batch_size = batch_size
    self.collate_fn = collate_fn

  def __iter__(self):
    for i in range(0, len(self.data), self.batch_size):
      yield self.collate_fn(self.data[i: i+self.batch_size])

# usually a custom function. Could also just return a list of dictionaries, etc. depending on how the data looks like
def collate_fn(batch): # given batch is a List
  max_pad_len = max(len(sample) for sample in batch)
  padded = [sample + [0] * (max_pad_len - len(sample)) for sample in batch]
  return padded
