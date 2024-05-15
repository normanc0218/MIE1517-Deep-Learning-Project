
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader
from torch.nn import functional as F
from tqdm import tqdm

from process_audio import LibriSpeechDataset, collate
from model import ASR, ASR1
from hyperparameters import hp

seed = 0
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True


def compute_validation_loss(net, criterion, dataloader):
    net.eval()
    losses = []
    for data, label, data_len, label_len in tqdm(dataloader):
        data, label, data_len, label_len = data.cuda(), label.cuda(), data_len.cuda(), label_len.cuda()
        out, _ = net(data)
        out = F.log_softmax(out, dim=2)
        out = out.transpose(0, 1)
        loss = criterion(out, label, data_len, label_len)
        losses += [loss.item()]

    return sum(losses) / len(losses)


if __name__ == "__main__":

    # Create wandb logger
    wandb.login()

    # Initialize model
    asr_model = ASR(hp["dropout"], hp["hidden_size"], hp["rnn_layers"], hp["cnn_layers"], hp["n_mels"])
    asr_model = asr_model.cuda()

    # Datasets
    train_dataset = LibriSpeechDataset("train")
    valid_dataset = LibriSpeechDataset("valid")

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=hp["batch_size"],
                              shuffle=True,
                              collate_fn=collate,
                              num_workers=3,
                              pin_memory=False)
    valid_loader = DataLoader(dataset=valid_dataset,
                                batch_size=hp["batch_size"],
                                shuffle=False,
                                collate_fn=collate,
                                num_workers=3,
                                pin_memory=False)

    # Train
    optimizer = optim.Adam(asr_model.parameters(), lr=hp["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=hp["lr_factor"], patience=hp["lr_patience"])
    criterion = nn.CTCLoss(blank=28, zero_infinity=True)

    min_valid_loss = 1e10
    min_train_loss = 1e10

    # with wandb.init(project="MIE1517", config=hp):
    #     wandb.watch(asr_model, log="all")

    for epoch in range(hp["epochs"]):

        asr_model = asr_model.train(True)

        train_losses = []
        for data, label, data_len, label_len in tqdm(train_loader, desc="Epoch {0} / {1}".format(epoch, hp["epochs"])):
            data, label, data_len, label_len = data.cuda(), label.cuda(), data_len.cuda(), label_len.cuda()
            # out, _ = asr_model(data)
            out, _ = asr_model(data)
            out = F.log_softmax(out, dim=2)
            out = out.transpose(0, 1) # CTCLoss takes batch_size as second dim
            loss = criterion(out, label, data_len, label_len)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_losses += [loss.item()]

        train_loss = sum(train_losses) / len(train_losses)
        avg_valid_loss = compute_validation_loss(asr_model, criterion, valid_loader)
        scheduler.step(avg_valid_loss)

        # Save checkpoint if valid loss is at minimum
        if avg_valid_loss < min_valid_loss:
            print("Saved valid checkpoint!")
            torch.save(asr_model.state_dict(), "/home/asblab2/sinarasi/mie1517/MIE1517-Project/Speech Recog/best_model.pth")
            min_valid_loss = avg_valid_loss
        if train_loss < min_train_loss:
            print("Saved train checkpoint!")
            torch.save(asr_model.state_dict(), "/home/asblab2/sinarasi/mie1517/MIE1517-Project/Speech Recog/mid_model.pth")
            min_train_loss = train_loss
        print("Losses:", train_loss, avg_valid_loss, optimizer.param_groups[0]['lr'])

            # wandb.log({"train_loss": train_loss,
            #            "valid_loss": avg_valid_loss}, step=epoch)

