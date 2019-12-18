# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F


def download_state_dict(model_name):

    base_url = "https://dl.fbaipublicfiles.com/librilight/CPC_checkpoints"
    return torch.hub.load_state_dict_from_url(f"{base_url}/{model_name}")


def load_cpc_features(state_dict):

    config = state_dict["config"]
    weights = state_dict["weights"]
    encoder = CPCEncoder(config["hiddenEncoder"])
    ar_net = CPCAR(config["hiddenEncoder"], config["hiddenGar"], False,
                   config["nLevelsGRU"])

    model = CPCModel(encoder, ar_net)
    model.load_state_dict(weights, strict=False)
    output = FeatureModule(model, False)
    output.config = config
    return output


def get_features_state_dict(feature_module):
    config = feature_module.config
    if config is None:
        raise ValueError("The input feature_module should have config defined")
    weights = feature_module.model.state_dict()
    return {"config": config, "weights": weights}


def build_feature_from_file(file_path, feature_maker, max_size_seq=64000):
    r"""
    Apply the featureMaker to the given file.
    Arguments:
        - file_path (FeatureModule): model to apply
        - file_path (string): path of the sequence to load
        - seq_norm (bool): if True, normalize the output along the time
                           dimension to get chunks of mean zero and var 1
        - max_size_seq (int): maximal size of a chunk
    Return:
        a torch vector of size 1 x Seq_size x Feature_dim
    """
    seq = torchaudio.load(file_path)[0]
    sizeSeq = seq.size(1)
    start = 0
    out = []
    while start < sizeSeq:
        if start + max_size_seq > sizeSeq:
            break
        end = min(sizeSeq, start + max_size_seq)
        subseq = (seq[:, start:end]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = feature_maker(subseq)
        out.append(features.detach().cpu())
        start += max_size_seq

    if start < sizeSeq:
        subseq = (seq[:, -max_size_seq:]).view(1, 1, -1).cuda(device=0)
        with torch.no_grad():
            features = feature_maker(subseq)
        df = subseq.size(2) // features.size(1)
        delta = (sizeSeq - start) // df
        out.append(features[:, -delta:].detach().cpu())

    out = torch.cat(out, dim=1)
    return out.view(out.size(1), out.size(2))

##############################################################################
# Minimal code to load a CPC checkpoint
##############################################################################


class ChannelNorm(nn.Module):

    def __init__(self,
                 numFeatures,
                 epsilon=1e-05,
                 affine=True):

        super(ChannelNorm, self).__init__()
        if affine:
            self.weight = nn.parameter.Parameter(
                torch.Tensor(1, numFeatures, 1))
            self.bias = nn.parameter.Parameter(torch.Tensor(1, numFeatures, 1))
        else:
            self.weight = None
            self.bias = None
        self.epsilon = epsilon
        self.p = 0
        self.affine = affine
        self.reset_parameters()

    def reset_parameters(self):
        if self.affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x):

        cumMean = x.mean(dim=1, keepdim=True)
        cumVar = x.var(dim=1, keepdim=True)
        x = (x - cumMean)*torch.rsqrt(cumVar + self.epsilon)

        if self.weight is not None:
            x = x * self.weight + self.bias
        return x


class CPCEncoder(nn.Module):

    def __init__(self,
                 sizeHidden=512):

        super(CPCEncoder, self).__init__()
        normLayer = ChannelNorm

        self.conv0 = nn.Conv1d(1, sizeHidden, 10, stride=5, padding=3)
        self.batchNorm0 = normLayer(sizeHidden)
        self.conv1 = nn.Conv1d(sizeHidden, sizeHidden, 8, stride=4, padding=2)
        self.batchNorm1 = normLayer(sizeHidden)
        self.conv2 = nn.Conv1d(sizeHidden, sizeHidden, 4,
                               stride=2, padding=1)
        self.batchNorm2 = normLayer(sizeHidden)
        self.conv3 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm3 = normLayer(sizeHidden)
        self.conv4 = nn.Conv1d(sizeHidden, sizeHidden, 4, stride=2, padding=1)
        self.batchNorm4 = normLayer(sizeHidden)
        self.DOWNSAMPLING = 160

    def getDimOutput(self):
        return self.conv4.out_channels

    def forward(self, x):
        x = F.relu(self.batchNorm0(self.conv0(x)))
        x = F.relu(self.batchNorm1(self.conv1(x)))
        x = F.relu(self.batchNorm2(self.conv2(x)))
        x = F.relu(self.batchNorm3(self.conv3(x)))
        x = F.relu(self.batchNorm4(self.conv4(x)))
        return x


class CPCAR(nn.Module):

    def __init__(self,
                 dimEncoded,
                 dimOutput,
                 keepHidden,
                 nLevelsGRU):

        super(CPCAR, self).__init__()
        self.baseNet = nn.LSTM(dimEncoded, dimOutput,
                               num_layers=nLevelsGRU, batch_first=True)
        self.hidden = None
        self.keepHidden = keepHidden

    def getDimOutput(self):
        return self.baseNet.hidden_size

    def forward(self, x):

        try:
            self.baseNet.flatten_parameters()
        except RuntimeError:
            pass
        x, h = self.baseNet(x, self.hidden)
        if self.keepHidden:
            if isinstance(h, tuple):
                self.hidden = tuple(x.detach() for x in h)
            else:
                self.hidden = h.detach()
        return x


class CPCModel(nn.Module):

    def __init__(self,
                 encoder,
                 AR):

        super(CPCModel, self).__init__()
        self.gEncoder = encoder
        self.gAR = AR

    def forward(self, batchData, label):
        encodedData = self.gEncoder(batchData).permute(0, 2, 1)
        cFeature = self.gAR(encodedData)
        return cFeature, encodedData, label


class FeatureModule(torch.nn.Module):
    r"""
    A simpler interface to handle CPC models. Useful for a smooth workflow when
    working with CPC trained features.
    """

    def __init__(self, featureMaker, get_encoded,
                 seq_norm=True):
        super(FeatureModule, self).__init__()
        self.get_encoded = get_encoded
        self.model = featureMaker
        self.seq_norm = seq_norm
        self.config = None

    def forward(self, batch_data):
        # Input Size : BatchSize x 1 x SeqSize
        # Feature size: BatchSize x SeqSize x ChannelSize
        if self.is_cuda:
            batch_data = batch_data.cuda()
        cFeature, encoded, _ = self.model(batch_data, None)
        if self.get_encoded:
            cFeature = encoded
        if self.seq_norm:
            mean = cFeature.mean(dim=1, keepdim=True)
            var = cFeature.var(dim=1, keepdim=True)
            cFeature = (cFeature - mean) / torch.sqrt(var + 1e-08)
        return cFeature

    def cuda(self):
        self.is_cuda = True
        super(FeatureModule, self).cuda()

    def cpu(self):
        self.is_cuda = False
        super(FeatureModule, self).cuda()

    def get_output_dim(self):
        if self.get_encoded:
            return self.config["hiddenEncoder"]
        return self.config["hiddenGar"]


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download model')
    parser.add_argument('model_name', type=str,
                        choices=["600h", "6kh", "60kh"])
    parser.add_argument('output', type=str)
    args = parser.parse_args()

    CPC_MODELS_NAMES = {"60kh": "60k_epoch4-d0f474de.pt",
                        "600h": "600h-bdd7ced6.pt",
                        "6kh":"6k_epoch30-9df0493c.pt"}
    state_dict = download_state_dict(CPC_MODELS_NAMES[args.model_name])
    torch.save(state_dict, args.output)
