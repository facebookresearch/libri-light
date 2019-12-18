# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torchaudio
from copy import deepcopy
import torch
import time
from pathlib import Path
from torch.utils.data import Dataset
from torch.multiprocessing import Pool


def load(path_item):
    seq_name = path_item.stem
    data = torchaudio.load(str(path_item))[0].view(1, -1)
    return seq_name, data


class SingleSequenceDataset(Dataset):

    def __init__(self,
                 pathDB,
                 seqNames,
                 phoneLabelsDict,
                 inDim=1,
                 transpose=True):
        """
        Args:
            - path (string): path to the training dataset
            - sizeWindow (int): size of the sliding window
            - seqNames (list): sequences to load
            - phoneLabels (dictionnary): if not None, a dictionnary with the
                                         following entries

                                         "step": size of a labelled window
                                         "$SEQ_NAME": list of phonem labels for
                                         the sequence $SEQ_NAME
        """
        self.seqNames = deepcopy(seqNames)
        self.pathDB = pathDB
        self.phoneLabelsDict = deepcopy(phoneLabelsDict)
        self.inDim = inDim
        self.transpose = transpose
        self.loadSeqs()

    def loadSeqs(self):

        # Labels
        self.seqOffset = [0]
        self.phoneLabels = []
        self.phoneOffsets = [0]
        self.data = []
        self.maxSize = 0
        self.maxSizePhone = 0

        # Data

        nprocess = min(30, len(self.seqNames))

        start_time = time.time()
        to_load = [Path(self.pathDB) / x for _, x in self.seqNames]

        with Pool(nprocess) as p:
            poolData = p.map(load, to_load)

        tmpData = []
        poolData.sort()

        totSize = 0
        minSizePhone = 1000000
        for seqName, seq in poolData:
            self.phoneLabels += self.phoneLabelsDict[seqName]
            self.phoneOffsets.append(len(self.phoneLabels))
            self.maxSizePhone = max(self.maxSizePhone,
                                    len(self.phoneLabelsDict[seqName]))
            minSizePhone = min(minSizePhone, len(
                self.phoneLabelsDict[seqName]))
            sizeSeq = seq.size(1)
            self.maxSize = max(self.maxSize, sizeSeq)
            totSize += sizeSeq
            tmpData.append(seq)
            self.seqOffset.append(self.seqOffset[-1] + sizeSeq)
            del seq
        self.data = torch.cat(tmpData, dim=1)
        self.phoneLabels = torch.tensor(self.phoneLabels, dtype=torch.long)
        print(f'Loaded {len(self.phoneOffsets)} sequences '
              f'in {time.time() - start_time:.2f} seconds')
        print(f'maxSizeSeq : {self.maxSize}')
        print(f'maxSizePhone : {self.maxSizePhone}')
        print(f"minSizePhone : {minSizePhone}")
        print(f'Total size dataset {totSize / (16000 * 3600)} hours')

    def __getitem__(self, idx):

        offsetStart = self.seqOffset[idx]
        offsetEnd = self.seqOffset[idx+1]
        offsetPhoneStart = self.phoneOffsets[idx]
        offsetPhoneEnd = self.phoneOffsets[idx + 1]

        sizeSeq = int(offsetEnd - offsetStart)
        sizePhone = int(offsetPhoneEnd - offsetPhoneStart)

        outSeq = torch.zeros((self.inDim, self.maxSize))
        outPhone = torch.zeros((self.maxSizePhone))

        outSeq[:, :sizeSeq] = self.data[:, offsetStart:offsetEnd]
        outPhone[:sizePhone] = self.phoneLabels[offsetPhoneStart:offsetPhoneEnd]

        return outSeq,  torch.tensor([sizeSeq], dtype=torch.long), outPhone.long(),  torch.tensor([sizePhone], dtype=torch.long)

    def __len__(self):
        return len(self.seqOffset) - 1


class CTCPhoneCriterion(torch.nn.Module):

    def __init__(self, dimEncoder, nPhones, LSTM=False, sizeKernel=8,
                 seqNorm=False, dropout=False, reduction='mean'):

        super(CTCPhoneCriterion, self).__init__()
        self.seqNorm = seqNorm
        self.epsilon = 1e-8
        self.dropout = torch.nn.Dropout2d(
            p=0.5, inplace=False) if dropout else None
        self.conv1 = torch.nn.LSTM(dimEncoder, dimEncoder,
                                   num_layers=1, batch_first=True)
        self.PhoneCriterionClassifier = torch.nn.Conv1d(
            dimEncoder, nPhones + 1, sizeKernel, stride=sizeKernel // 2)
        self.lossCriterion = torch.nn.CTCLoss(blank=nPhones,
                                              reduction=reduction,
                                              zero_infinity=True)
        self.relu = torch.nn.ReLU()
        self.BLANK_LABEL = nPhones
        self.useLSTM = LSTM

    def getPrediction(self, cFeature):
        B, S, H = cFeature.size()
        if self.seqNorm:
            m = cFeature.mean(dim=1, keepdim=True)
            v = cFeature.var(dim=1, keepdim=True)
            cFeature = (cFeature - m) / torch.sqrt(v + self.epsilon)
        if self.useLSTM:
            cFeature = self.conv1(cFeature)[0]

        cFeature = cFeature.permute(0, 2, 1)

        if self.dropout is not None:
            cFeature = self.dropout(cFeature)

        return self.PhoneCriterionClassifier(cFeature).permute(0, 2, 1)

    def forward(self, cFeature, featureSize, label, labelSize):

        # cFeature.size() : batchSize x seq Size x hidden size
        B, S, H = cFeature.size()
        predictions = self.getPrediction(cFeature)
        featureSize /= 4
        predictions = cutData(predictions, featureSize)
        featureSize = torch.clamp(featureSize, max=predictions.size(1))
        label = cutData(label, labelSize)
        if labelSize.min() <= 0:
            print(label, labelSize)
        predictions = torch.nn.functional.log_softmax(predictions, dim=2)
        predictions = predictions.permute(1, 0, 2)
        loss = self.lossCriterion(predictions, label,
                                  featureSize, labelSize).view(1, -1)

        if torch.isinf(loss).sum() > 0 or torch.isnan(loss).sum() > 0:
            loss = 0

        return loss


def cutData(seq, sizeSeq):
    maxSeq = sizeSeq.max()
    seq = seq[:, :maxSeq]
    return seq


def prepareData(data):
    seq, sizeSeq, phone, sizePhone = data
    seq = seq.cuda(non_blocking=True)
    phone = phone.cuda(non_blocking=True)
    sizeSeq = sizeSeq.cuda(non_blocking=True).view(-1)
    sizePhone = sizePhone.cuda(non_blocking=True).view(-1)

    seq = cutData(seq, sizeSeq)

    return seq, sizeSeq, phone, sizePhone


def trainStep(trainLoader,
              model,
              criterion,
              optimizer,
              downsamplingFactor):

    if model.optimize:
        model.train()

    criterion.train()
    avg_loss = 0
    nItems = 0

    for data in trainLoader:
        optimizer.zero_grad()
        seq, sizeSeq, phone, sizePhone = prepareData(data)
        c_feature = model(seq)
        sizeSeq = sizeSeq / downsamplingFactor
        loss = criterion(c_feature, sizeSeq, phone, sizePhone)
        loss.mean().backward()

        avg_loss += loss.mean().item()
        nItems += 1
        optimizer.step()

    return avg_loss / nItems


def valStep(valLoader,
            model,
            criterion,
            downsamplingFactor):

    model.eval()
    criterion.eval()
    avg_loss = 0
    nItems = 0

    for data in valLoader:
        with torch.no_grad():
            seq, sizeSeq, phone, sizePhone = prepareData(data)
            c_feature = model(seq)
            sizeSeq = sizeSeq / downsamplingFactor
            loss = criterion(c_feature, sizeSeq, phone, sizePhone)
            avg_loss += loss.mean().item()
            nItems += 1

    return avg_loss / nItems
