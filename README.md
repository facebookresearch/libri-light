# Libri-Light: a (large) dataset for ASR with limited or no supervision


This repo contains code associated with the Libri-light dataset which can be downloaded HERE. It contains code for data preparation, pretrained models and evaluation.


    data_preparation/         # dowload code; VAD and SNR code; json generation; stats; data segmentation
    eval/                     # ABX, PER, WER (evaluation metrics on dev-clean, dev-other, test-clean, test-other)
    TDS/                      # code, pretrained models, examples

To install:

     git clone https://github.com/facebookresearch/libri-light



## Goals and structure

Libri-light offers 60+ k hours of unlabelled speech, a small training set for limited supervision (10h, 1h or 10 minutes of labelled speech), and a common set of metrics to evaluated three settings:

  1. the unsupervised/zero-resource setting. Here, models are trained only on unlabelleds speech and attempt to construct 'good' speech representations. They are evaluated with the ABX metric.
  2. the semi-supervised setting. Here, models are trained with the limited supervision dataset and exploit the unlabelled in various ways (as pretraining, to get pseudo-labels, etc). The models are evaluated using either PER or WER.
  3. the distant supervision setting. Here, models can use additional unaligned text to build a decoder. These models are evaluated using WER.


## Documentation

To use librilight, first go in `data_preparation/` and follow the instructions in the section "Download the data".
You can then go in `eval/` and run ABX, PER and WER evaluations on pretrained CPC models. 
Finally, you'll find in `TDS` the wav2letter experiments described in the paper.




## Citing

    @misc{librilight,
        author = {Kahn, J and Rivière, M. and Zheng, W., and Kharitonov, E., and Xu, Q. and
         Mazaré, P.E. and Karadayi, J. and Liptchinsky, V. and Collobert, R. and Fuegen, C. and
         Likhomanenko, T. and Synnaeve, G. and Joulin, A. and Mohamed, A. and Dupoux, E.},
        title = {{Libri-light: a (large) dataset for ASR with limited or no supervision}},
        year = {2019},
        publisher = {GitHub},
        journal = {ArXiv repository},
        howpublished = {\url{https://GitHub.com/FacebookResearch/Libri-light}},
    }

## License

The Libri-light code is released under the MIT license. See LICENSE for additional details.
