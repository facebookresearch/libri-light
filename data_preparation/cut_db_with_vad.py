import soundfile as sf
import numpy as np
import progressbar
import argparse
from pathlib import Path
from multiprocessing import Process, Lock, Value
import json
import os
import sys


def load_lst(path):

    with open(path, 'r') as file:
        data = file.readlines()

    return [x.replace('\n', '').split() for x in data]


def get_books_from_lst(lst_data):
    path_seqs = [Path(x[1]) for x in lst_data]
    return list({str(x.parent) for x in path_seqs})


def loadVAD(pathVAD):

    with open(pathVAD, 'r') as file:
        data = file.readlines()

    assert(len(data) == 1 )
    vadtxt = data[0].replace('\n', '')
    vad = vadtxt.split()
    if vad[-1] == '':
        vad = vad[:-1]

    return [1.0 - float(x) for x in vad]


def cutWithVAD(seq, vad, threshold, stepVAD):

    size = seq.shape[0]
    assert(stepVAD * (len(vad) -1) <= size)

    out = []
    lastIndex=0
    for index, vadStatus in enumerate(vad):

        if vadStatus < threshold:
            if lastIndex < index:
                upIndex = index if index == len(vad) -1 else index+1
                out.append((lastIndex * stepVAD, upIndex * stepVAD))
            lastIndex=index+1

    if lastIndex < len(vad):
        out.append((lastIndex * stepVAD, size))
    return out


def greedyMerge(cuts, targetSize, sizeMultiplier):

    out = []
    currSize = 0
    currBitSize=0
    lastIndex = 0
    for index, item in enumerate(cuts):
        minStep, maxStep = item
        currBitSize+=(maxStep - minStep)
        currSize+= (maxStep - minStep) * sizeMultiplier
        if currSize > targetSize:
            out.append((currBitSize, cuts[lastIndex:index+1]))
            lastIndex= index + 1
            currSize=0
            currBitSize=0

    if lastIndex < len(cuts):
        out.append((currBitSize, cuts[lastIndex:]))

    return out


def saveWithTimeStamps(seq, timeStamps, dirOut, nameOut, format, samplerate):

    for index, cutsBlock in enumerate(timeStamps):

        size, cuts = cutsBlock
        locSeq = np.empty((size,), dtype=seq.dtype)
        i=0
        for indexLow, indexUp in cuts:
            locSize = indexUp - indexLow
            locSeq[i: i+locSize] = seq[indexLow:indexUp]
            i+=locSize
        fullName = f"{nameOut}_{index:04}{format}"
        fullPathOut = os.path.join(dirOut, fullName)

        sf.write(fullPathOut, locSeq, samplerate)


def cutSequence(pathSeq, pathVAD, pathOut, chunkSize,nameOut, formatOut=".wav",
                targetSize=10, threshold=0.001):
    vad = loadVAD(pathVAD)
    data, samplerate = sf.read(pathSeq)
    assert(len(data.shape) == 1)
    assert(samplerate == 16000)

    step = int(samplerate * chunkSize)
    vadCuts = cutWithVAD(data,vad, threshold, step)
    mergeTimeStamps = greedyMerge(vadCuts, targetSize, 1.0 / samplerate)
    saveWithTimeStamps(data, mergeTimeStamps, pathOut,
                       nameOut, formatOut, samplerate)


def cutBook(pathDIRBook, pathDIRSpeakerData, pathOutDB,
            extAudio='.wav', extVAD='.vad', chunkSize=0.08, targetSize=10,
            lock=None):

    bookName = os.path.basename(os.path.normpath(pathDIRBook))
    speakerDataName = bookName+'_speaker_data.json'
    bookMetadataName = bookName+'_metadata.json'
    # speakerDataName = bookName.replace('librivox_wav',
    #                                    'librivox_64kb_mp3_speaker_data.json')
    # bookMetadataName = bookName.replace('librivox_wav',
    #                                     'librivox_64kb_mp3_metadata.json')
    speakerMetadataPath = os.path.join(pathDIRSpeakerData, speakerDataName)
    bookMetadataPath = os.path.join(pathDIRSpeakerData, bookMetadataName)

    if not os.path.isfile(speakerMetadataPath):
        return False

    if not os.path.isfile(bookMetadataPath):
        return False

    with open(speakerMetadataPath, 'rb') as file:
        speakerData = json.load(file)

    with open(bookMetadataPath, 'rb') as file:
        bookMetadata = json.load(file)

    bookID = int(bookMetadata["id"])

    if speakerData["names"] is None:
        return False

    for bookIndex, bookName in enumerate(speakerData["names"]):

        speakerIDs = speakerData["readers"][bookIndex]
        if speakerIDs is None or len(speakerIDs) > 1 or speakerIDs[0] is None:
            continue

        speakerID = speakerIDs[0]
        speakerDIR = os.path.join(pathOutDB, speakerID)
        if lock is not None:
            lock.acquire()
        Path(speakerDIR).mkdir(exist_ok=True)
        chapterDIR = os.path.join(speakerDIR, str(bookIndex))
        Path(chapterDIR).mkdir(exist_ok=True)

        if lock is not None:
            lock.release()

        bookName = bookName.replace('_128kb', '')

        nameOut = f"{speakerID}_{bookID}_{bookIndex}"
        pathSeq = os.path.join(pathDIRBook, f"{bookName}.wav")

        if not os.path.isfile(pathSeq):
            pathSeq = os.path.join(pathDIRBook, f"{bookName}_64kb.wav")

        if not os.path.isfile(pathSeq):
            print(pathSeq)
            continue

        pathVAD = str(Path(pathSeq).with_suffix('.vad'))

        if not os.path.isfile(pathVAD):
            print(pathVAD)
            continue

        cutSequence(pathSeq, pathVAD, chapterDIR, chunkSize, nameOut,
                    formatOut=".wav", targetSize=targetSize, threshold=0.001)


def cut_db(book_data, pathMetadata, pathOut, targetSize=10,
           nProcess=30):

    n_books = len(book_data)

    print(f"{n_books} books detected")
    print(f"Launching {nProcess} processes")
    pbar = progressbar.ProgressBar(maxval=n_books)
    pbar.start()

    def processStack(v, l, indexStart, indexEnd):
        for index, pathBook in enumerate(book_data[indexStart:indexEnd]):
            cutBook(pathBook, pathMetadata, pathOut, targetSize=targetSize,
                    lock=l)
            l.acquire()
            v.value+=1
            pbar.update(v.value)
            l.release()

    stack = []
    sizeSlice = n_books // nProcess
    lock = Lock()
    v = Value('i', 0)
    for process in range(nProcess):
        indexStart = sizeSlice * process
        indexEnd = indexStart + sizeSlice if process < nProcess - 1 else n_books
        p = Process(target=processStack, args=(v, lock, indexStart,indexEnd))
        p.start()
        stack.append(p)

    for process in stack:
        p.join()

    pbar.finish()


def parse_args():

    parser = argparse.ArgumentParser(description="Cut a dataset in small "
                                     "sequences using VAD files")
    parser.add_argument('path_lst', type=str,
                        help="Path to the lst file used to build the vad data")
    parser.add_argument('out_dir', type=str, default=None,
                        help="Path to the output directory")
    parser.add_argument('--path_metadata', type=str, default=None,
                        help="If different from path_db, path to the directory "
                             "containing the metadata")
    parser.add_argument('--target_time', type=int, default=60,
                        help="Target time, in seconds of each output sequence"
                             "(default is 10)")
    return parser


if __name__ == "__main__":
    argv = sys.argv[1:]
    parser = parse_args()
    args = parser.parse_args(argv)

    if args.path_metadata is None:
        args.path_metadata = args.path_db

    lst_data = load_lst(args.path_lst)
    book_data = get_books_from_lst(lst_data)

    Path.mkdir(Path(args.out_dir), exist_ok=True)
    cut_db(book_data, args.path_metadata, args.out_dir,
           targetSize=args.target_time)

    # pathDB="/checkpoint/jacobkahn/librivox/wav-ffmpeg/"
    # pathDIRMetaData="/checkpoint/mriviere/LibriVox/"
    # pathOut = "/checkpoint/antares/librivox"
