# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from copy import deepcopy
import torch
import progressbar
import math
import numpy as np
from torch.multiprocessing import Lock, Manager
from .PER_src import per_operator


def cut_data(seq, sizeSeq):
    maxSeq = sizeSeq.max()
    seq = seq[:, :maxSeq]
    return seq


def beam_search(score_preds, nKeep, blankLabel):

    T, P = score_preds.shape
    beams = set([''])
    pb_t_1 = {"": 1}
    pnb_t_1 = {"": 0}

    def getLastNumber(b):
        return int(b.split(',')[-1])

    for t in range(T):

        nextBeams = set()
        pb_t = {}
        pnb_t = {}
        for i_beam, b in enumerate(beams):
            if b not in pb_t:
                pb_t[b] = 0
                pnb_t[b] = 0

            if len(b) > 0:
                pnb_t[b] += pnb_t_1[b] * score_preds[t, getLastNumber(b)]
            pb_t[b] = (pnb_t_1[b] + pb_t_1[b]) * score_preds[t, blankLabel]
            nextBeams.add(b)

            for c in range(P):
                if c == blankLabel:
                    continue

                b_ = b + "," + str(c)
                if b_ not in pb_t:
                    pb_t[b_] = 0
                    pnb_t[b_] = 0

                if b != "" and getLastNumber(b) == c:
                    pnb_t[b_] += pb_t_1[b] * score_preds[t, c]
                else:
                    pnb_t[b_] += (pb_t_1[b] + pnb_t_1[b]) * score_preds[t, c]
                nextBeams.add(b_)

        allPreds = [(pb_t[b] + pnb_t[b], b) for b in nextBeams]
        allPreds.sort(reverse=True)

        beams = [x[1] for x in allPreds[:nKeep]]
        pb_t_1 = deepcopy(pb_t)
        pnb_t_1 = deepcopy(pnb_t)

    output = []
    for score, x in allPreds[:nKeep]:
        output.append((score, [int(y) for y in x.split(',') if len(y) > 0]))
    return output


def get_seq_PER(seqLabels, detectedLabels):
    return per_operator.needleman_wunsch_align_score(seqLabels, detectedLabels,
                                                     -1, -1, 0,
                                                     normalize=True)


def prepare_data(data):
    seq, sizeSeq, phone, sizePhone = data
    seq = seq.cuda(non_blocking=True)
    phone = phone.cuda(non_blocking=True)
    sizeSeq = sizeSeq.cuda(non_blocking=True).view(-1)
    sizePhone = sizePhone.cuda(non_blocking=True).view(-1)

    seq = cut_data(seq, sizeSeq)

    return seq, sizeSeq, phone, sizePhone


def get_local_per(pool, mutex, p_, gt_seq, BLANK_LABEL):
    predSeq = np.array(beam_search(p_, 10, BLANK_LABEL)[0][1], dtype=np.int32)
    per = get_seq_PER(gt_seq, predSeq)
    mutex.acquire()
    pool.append(per)
    mutex.release()


def per_step(valLoader,
             model,
             criterion,
             downsamplingFactor):

    model.eval()
    criterion.eval()

    avgPER = 0
    varPER = 0
    nItems = 0

    print("Starting the PER computation through beam search")
    bar = progressbar.ProgressBar(maxval=len(valLoader))
    bar.start()

    for index, data in enumerate(valLoader):

        bar.update(index)

        with torch.no_grad():
            seq, sizeSeq, phone, sizePhone = prepare_data(data)
            c_feature = model(seq)
            sizeSeq = sizeSeq / downsamplingFactor
            predictions = torch.nn.functional.softmax(criterion.getPrediction(c_feature),
                                                      dim=2).cpu()
            c_feature = c_feature
            phone = phone.cpu()
            sizeSeq = sizeSeq.cpu()
            sizePhone = sizePhone.cpu()

            mutex = Lock()
            manager = Manager()
            poolData = manager.list()

            processes = []
            for b in range(sizeSeq.size(0)):
                l_ = min(sizeSeq[b] // 4, predictions.size(1))
                s_ = sizePhone[b]
                p = torch.multiprocessing.Process(target=get_local_per,
                                                  args=(poolData, mutex, predictions[b, :l_].view(l_, -1).numpy(),
                                                        phone[b, :s_].view(-1).numpy().astype(np.int32), criterion.BLANK_LABEL))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()

            avgPER += sum([x for x in poolData])
            varPER += sum([x*x for x in poolData])
            nItems += len(poolData)

    bar.finish()

    avgPER /= nItems
    varPER /= nItems

    varPER -= avgPER**2
    print(f"Average PER {avgPER}")
    print(f"Standard deviation PER {math.sqrt(varPER)}")
