import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, word_vocab_size, case_vocab_size, char_vocab_size,
                 word_embeddings_dim, case_embeddings_dim, char_embeddings_dim,
                 lstm_hidden_dim, char_out_dim, tag_set_size,
                 word_embeddings, case_embeddings):
        super().__init__()

        self.word_embeddings = nn.Embedding(word_vocab_size, word_embeddings_dim)
        self.word_embeddings.weight = nn.Parameter(torch.FloatTensor(word_embeddings))

        self.case_embeddings = nn.Embedding(case_vocab_size, case_embeddings_dim)
        self.case_embeddings.weight = nn.Parameter(torch.FloatTensor(case_embeddings))

        self.char_embeddings = nn.Embedding(char_vocab_size, char_embeddings_dim)

        # self.lstm = nn.LSTM(word_embeddings_dim + case_embeddings_dim + char_out_dim,
        #                     lstm_hidden_dim, bidirectional=True, batch_first=True)

        self.lstm = nn.LSTM(word_embeddings_dim + case_embeddings_dim,
                            lstm_hidden_dim, bidirectional=True, batch_first=True)

        self.conv = nn.Conv2d(1, char_out_dim, (3, char_embeddings_dim))

        self.hidden = nn.Linear(2 * lstm_hidden_dim, lstm_hidden_dim)
        self.out = nn.Linear(lstm_hidden_dim, tag_set_size)

        self.dropout = nn.Dropout(0.4)

        self.elu = nn.ELU()

        self.char_out_dim = char_out_dim

    def forward(self, data):
        word_embeddings = self.word_embeddings(data['word'])

        case_embeddings = self.case_embeddings(data['case'])

        # char_embeddings = self.char_embeddings(data['char'])
        # batch_size, max_sentence_length, max_word_length, char_embeddings_dim = char_embeddings.shape
        # char_embeddings = char_embeddings.view(-1, 1, max_word_length, char_embeddings_dim)
        # char_embeddings = self.conv(char_embeddings)
        # char_embeddings = char_embeddings.view(batch_size, max_sentence_length, self.char_out_dim, -1)
        # char_embeddings = char_embeddings.max(-1)[0]

        # embeddings = torch.cat((word_embeddings, case_embeddings, char_embeddings), -1)
        embeddings = torch.cat((word_embeddings, case_embeddings), -1)
        embeddings = self.dropout(embeddings)

        output, _ = self.lstm(embeddings)

        output = self.hidden(output)
        output = self.elu(output)
        out = self.out(output)

        return out
