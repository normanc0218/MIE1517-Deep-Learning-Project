import torch, torchaudio
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model import ASR
from hyperparameters import hp
from process_audio import LibriSpeechDataset, collate

import time
import sounddevice as sd


class GreedyCTCDecoder:

    def __init__(self):
        self.text_to_int = {"'": 0, " ": 1, "a": 2, "b": 3, "c": 4,
                            "d": 5, "e": 6, "f": 7, "g": 8, "h": 9,
                            "i": 10, "j": 11, "k": 12, "l": 13, "m": 14,
                            "n": 15, "o": 16, "p": 17, "q": 18, "r": 19,
                            "s": 20, "t": 21, "u": 22, "v": 23, "w": 24,
                            "x": 25, "y": 26, "z": 27}
        self.int_to_text = {v: k for k, v in self.text_to_int.items()}
        self.blank = 28

    def __call__(self, emission, sentence, length):

        # Compute prediction
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i.item() for i in indices if i.item() != self.blank]
        joined = "".join([self.int_to_text[i] for i in indices])

        # Compute actual
        truth = ""
        sentence = sentence.squeeze()
        for i in range(length):
            truth += self.int_to_text[int(sentence[i])]

        return joined, truth


def record_audio(duration, sample_rate=16000):
    print("Recording audio...")
    audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()
    print("Recording finished.")
    return torch.from_numpy(audio_data.flatten())


if __name__ == "__main__":

    # Record data
    # audio_tensor = record_audio(5).unsqueeze(0)
    # torchaudio.save("../../myaudio.wav", audio_tensor, sample_rate=16000)

    # Load data from file
    data_piece = torchaudio.load("myaudio.wav")[0]
    label_piece = "i am looking for a book"
    one_dataset = LibriSpeechDataset("one", (data_piece, label_piece))
    loader = DataLoader(dataset=one_dataset, batch_size=1, shuffle=True, collate_fn=collate)

    # Model
    model = ASR(hp["dropout"], hp["hidden_size"], hp["rnn_layers"], hp["cnn_layers"], hp["n_mels"])
    model = model.cuda()
    checkpoint = torch.load("mid_model_lib.pth")
    model.load_state_dict(checkpoint)
    model.eval()

    # Output sample predictions
    greedy_decoder = GreedyCTCDecoder()
    for spectograms, labels, data_lengths, label_lengths in iter(loader):
        spectograms, labels = spectograms.cuda(), labels.cuda()
        output = model(spectograms)
        output, _ = F.log_softmax(output, dim=2)
        output = output.squeeze(0)
        predicted = greedy_decoder(output, labels, label_lengths)
        print("Predicted:", predicted[0])
        print("Actual:", predicted[1])
        print("_____________________________________")
        time.sleep(1)