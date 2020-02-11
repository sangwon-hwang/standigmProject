import torch
from torch import nn
from gentrl.tokenizer import encode, get_vocab_size

class CNNEncoder(nn.Module):
    def __init__(self, hidden_size=256, latent_size=50):
        super(CNNEncoder, self).__init__()

        self.embs = nn.Embedding(get_vocab_size(), hidden_size)
        self.cnn = nn.Conv1d(50,50,1)

        self.final_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
            nn.Linear(hidden_size, 2 * latent_size))

    def encode(self, sm_list):

        tokens, lens = encode(sm_list)
        to_feed = tokens.transpose(1, 0).to(self.embs.weight.device)

        outputs = self.cnn(self.embs(to_feed))
        outputs = self.cnn(outputs)

        outputs = outputs[lens, torch.arange(len(lens))]

        return self.final_mlp(outputs)
