import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, word_vocab_size, case_vocab_size, word_embeddings_dim, case_embeddings_dim,
                 lstm_hidden_dim, tag_set_size, word_embeddings, case_embeddings):
        super().__init__()

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embeddings_dim)
        self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(word_embeddings))

        self.case_embeddings = nn.Embedding(case_vocab_size, case_embeddings_dim)
        self.case_embeddings.weight = nn.Parameter(torch.FloatTensor(case_embeddings))

        self.lstm = nn.LSTM(word_embeddings_dim + case_embeddings_dim, lstm_hidden_dim,
                            bidirectional=True, batch_first=True)

        self.out = nn.Linear(2 * lstm_hidden_dim, tag_set_size)

        self.dropout = nn.Dropout(0.4)

    def forward(self, data):
        word_embeddings = self.word_embeddings(data['word'])
        case_embeddings = self.case_embeddings(data['case'])
        embeddings = torch.cat((word_embeddings, case_embeddings), -1)

        embeddings = self.dropout(embeddings)
        output, _ = self.lstm(embeddings)
        output = self.dropout(output)

        out = self.out(output)

        return out
