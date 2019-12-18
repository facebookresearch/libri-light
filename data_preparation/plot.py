# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def plot_hist(seq, nBins, pathOut, y_label="", title="",
              x_label="", normalized=True, y_scale=None, x_scale=None):

    if isinstance(seq, list):
        seq = np.array(seq)

    counts, bins = np.histogram(seq, bins=nBins)
    if normalized:
        counts = counts / np.sum(counts)
    plt.style.use('seaborn')
    plt.clf()
    plt.hist(bins[:-1], bins, weights=counts)
    plt.ylabel(y_label)
    plt.xlabel(x_label)
    plt.title(title)
    if y_scale is not None:
        plt.yscale(y_scale)
    if x_scale is not None:
        plt.yscale(x_scale)
    plt.tight_layout()
    plt.savefig(pathOut)


def plot_scatter(seqs, xLabel, pathOut, x_label="", y_label="", title=""):
    plt.clf()
    for i in range(seqs.shape[0]):
        plt.scatter(xLabel, seqs[i])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(pathOut)


def plot_seq(seqs, xLabel, pathOut, x_label="", y_label="", title="",
             xscale="linear", yscale="linear", legend=None):
    plt.clf()
    for i in range(seqs.shape[0]):
        plt.plot(xLabel, seqs[i])
    plt.xscale(xscale)
    plt.yscale(yscale)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    if legend is not None:
        plt.gca().legend(legend)
    plt.tight_layout()
    plt.savefig(pathOut)


def plot_pie(data, pathOut, title=""):

    labels = list(data.keys())
    sizes = [data[x] for x in labels]

    plt.clf()
    plt.style.use('classic')
    patches, texts, _ = plt.pie(sizes, autopct=lambda p: '{:.0f}'.format(p * sum(sizes) / 100),
                                shadow=False, startangle=90, pctdistance=1.1)
    #plt.axes([0.3, 0.3, .5, .5])
    plt.legend(patches, labels, loc='lower right',
               fontsize=8)
    plt.tight_layout()
    plt.title(title)
    plt.savefig(pathOut, bbox_inches='tight')
