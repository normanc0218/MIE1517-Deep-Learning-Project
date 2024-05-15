import torch, torchaudio
import torch.nn as nn
from torch.utils.data import Dataset

from hyperparameters import hp


class LibriSpeechOne(Dataset):
    def __init__(self, audio, label):
        self.audio = audio
        self.label = label

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.audio, None, self.label, None, None, None


class LibriSpeechDataset(Dataset):

    def __init__(self, dataset_type, data=None):

        self.audio_transform = nn.Sequential(
            torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_mels=128),
            torchaudio.transforms.FrequencyMasking(freq_mask_param=30),
            torchaudio.transforms.TimeMasking(time_mask_param=100)
        )

        self.dataset_dir = "/home/asblab2/sinarasi/mie1517/new_code/data"
        if dataset_type == "train":
            self.dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_dir, url="train-clean-100", download=True)
        elif dataset_type == "valid":
            self.dataset = torchaudio.datasets.LIBRISPEECH(self.dataset_dir, url="test-clean", download=True)
        elif dataset_type == "one":
            self.dataset = LibriSpeechOne(*data)
        else:
            raise Exception("Invalid dataset type!")


        self.text_to_int = {"'": 0, " ": 1, "a": 2, "b": 3, "c": 4,
                            "d": 5, "e": 6, "f": 7, "g": 8, "h": 9,
                            "i": 10, "j": 11, "k": 12, "l": 13, "m": 14,
                            "n": 15, "o": 16, "p": 17, "q": 18, "r": 19,
                            "s": 20, "t": 21, "u": 22, "v": 23, "w": 24,
                            "x": 25, "y": 26, "z": 27}
        self.int_to_text = {v: k for k, v in self.text_to_int.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        data = self.dataset[index]
        audio, _, sentence, _, _, _ = data
        spectogram = self.audio_transform(audio).squeeze(0).transpose(0, 1)
        label = [self.text_to_int[s] for s in sentence.lower()]
        spectogram_length = spectogram.shape[0] // 2
        label_length = len(label)
        return spectogram, label, spectogram_length, label_length


def collate(data):
    """
    Pad spectograms and labels within the batch to the same length.
    """
    spectograms, labels, spectogram_lengths, label_lengths = [], [], [], []
    for spectogram, label, spectogram_length, label_length in data:
        spectograms += [torch.Tensor(spectogram)]
        labels += [torch.Tensor(label)]
        spectogram_lengths += [spectogram_length]
        label_lengths += [label_length]
    spectograms = nn.utils.rnn.pad_sequence(spectograms, batch_first=True).transpose(1, 2)
    labels = nn.utils.rnn.pad_sequence(labels, batch_first=True)

    return spectograms, labels, torch.tensor(spectogram_lengths), torch.tensor(label_lengths)


if __name__ == "__main__":
    dataset = LibriSpeechDataset("train")
    print(dataset[0])
    print("Dataset Length:", len(dataset))