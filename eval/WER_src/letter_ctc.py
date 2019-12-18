# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch


def cut_data(seq, seq_len):
    max_len = seq_len.max()
    seq = seq[:, :max_len]
    return seq


class LetterClassifier(torch.nn.Module):

    def __init__(self, feature_maker, dim_encoder, n_letters, kernel_size=8, p_dropout=0.0):
        super().__init__()
        self.feature_maker = feature_maker
        self.feature_maker.eval()
        self.dropout = torch.nn.Dropout2d(p=p_dropout)
        self.lstm = torch.nn.LSTM(dim_encoder, dim_encoder // 2, bidirectional=True,
                                  num_layers=1, batch_first=True)
        self.classifier = torch.nn.Conv1d(
            dim_encoder, n_letters + 1, kernel_size, stride=kernel_size // 2)

    def forward(self, raw):
        with torch.no_grad():
            features = self.feature_maker(raw)

        self.lstm.flatten_parameters()
        x = self.lstm(features)[0]
        x = x.permute(0, 2, 1)
        x = self.dropout(x)
        return self.classifier(x).permute(0, 2, 1)


class CTCLetterCriterion(torch.nn.Module):

    def __init__(self, letter_classifier, n_letters):
        super().__init__()
        self.letter_classifier = letter_classifier
        self.loss = torch.nn.CTCLoss(blank=n_letters,
                                     zero_infinity=True)

    def forward(self, features, feature_size, label, label_size):
        predictions = self.letter_classifier(features)
        predictions = cut_data(predictions, feature_size)
        feature_size = torch.clamp(feature_size, max=predictions.size(1))
        label = cut_data(label, label_size)
        assert label_size.min() > 0
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        loss = self.loss(predictions, label, feature_size,
                         label_size).view(1, -1)

        assert not (torch.isinf(loss).any() or torch.isnan(loss).any())

        return loss
