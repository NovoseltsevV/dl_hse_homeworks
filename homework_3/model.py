import torch
from typing import Type
from torch import nn
from dataset import TextDataset


class LanguageModel(nn.Module):
    def __init__(self, dataset: TextDataset, embed_size: int = 256, hidden_size: int = 256,
                 rnn_type: Type = nn.RNN, rnn_layers: int = 1):
        """
        Model for text generation
        :param dataset: text data dataset (to extract vocab_size and max_length)
        :param embed_size: dimensionality of embeddings
        :param hidden_size: dimensionality of hidden state
        :param rnn_type: type of RNN layer (nn.RNN or nn.LSTM)
        :param rnn_layers: number of layers in RNN
        """
        super().__init__()
        self.dataset = dataset  # required for decoding during inference
        self.vocab_size = dataset.vocab_size
        self.max_length = dataset.max_length
        self.pad_id = dataset.pad_id
        self.bos_id = self.dataset.bos_id
        self.eos_id = self.dataset.eos_id

        self.embedding = nn.Embedding(self.vocab_size, embed_size, padding_idx=self.pad_id)
        self.rnn = rnn_type(embed_size, hidden_size, rnn_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, self.vocab_size)

    def forward(self, indices: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Compute forward pass through the model and
        return logits for the next token probabilities
        :param indices: LongTensor of encoded tokens of size (batch_size, length)
        :param lengths: LongTensor of lengths of size (batch_size, )
        :return: FloatTensor of logits of shape (batch_size, length, vocab_size)
        """
        emb = self.embedding(indices) # (batch_size, length, embed_size)
        packed_emb = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.rnn(packed_emb)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        logits = self.linear(output)
        return logits

    @torch.inference_mode()
    def inference(self, prefix: str = '', temp: float = 1.) -> str:
        """
        Generate new text with an optional prefix
        :param prefix: prefix to start generation
        :param temp: sampling temperature
        :return: generated text
        """
        self.eval()
        encoded_prefix = self.dataset.text2ids(prefix)
        generated = [self.bos_id] + encoded_prefix
        device = self.linear.weight.device
        input = torch.tensor(generated).unsqueeze(0).to(device)

        output, h = self.rnn(self.embedding(input))

        while len(generated) < self.max_length:
            logits = self.linear(output[:, -1])
            probs = nn.functional.softmax(logits / temp, dim=-1)
            new_token = torch.multinomial(probs, num_samples=1)
            generated.append(new_token.item())

            if new_token.item() == self.eos_id:
                break 
            output, h = self.rnn(self.embedding(new_token), h)
        
        generated = self.dataset.ids2text(generated)
        return generated
