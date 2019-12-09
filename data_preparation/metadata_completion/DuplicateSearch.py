# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import json
import progressbar


def getSameAuthorGroups(listMetadata, pathDIR):

    output = {}
    nEmpty = 0
    nSeverals = 0
    for metadata_name in listMetadata:

        fullPath = os.path.join(pathDIR, metadata_name)
        with open(fullPath, 'rb') as file:
            authorsData = json.load(file)['authors']

        if len(authorsData) == 0:
            authorIDs = [-1]
            nEmpty += 1
        else:
            authorIDs = []
            for author in authorsData:
                id = author["id"]
                if id is None:
                    id = -1
                else:
                    id = int(id)
                authorIDs.append(id)

        if len(authorIDs) > 1:
            nSeverals += 1

        for id in authorIDs:
            if id not in output:
                output[id] = set()
            output[id].add(metadata_name)

    print(f"{nEmpty} books without author, {nSeverals} with several authors")
    return output


def getBaseStringData(in_str):

    in_str = in_str.lower()
    tmp = in_str.split()
    out = []

    for word in tmp:
        word = ''.join([char for char in word if char.isalnum()])
        if len(word) == 0:
            continue
        out.append(word)

    return out


def getTitleSimilarityScore(title1, title2):

    title1 = set(getBaseStringData(title1))
    title2 = set(getBaseStringData(title2))

    nCommon = len(title1.intersection(title2))
    nUnion = len(title1.union(title2))

    if nUnion == 0:
        return 0

    return nCommon / nUnion


def getBaseTitle(title):

    in_str = title.lower()

    labelWords = ["dramatic reading", "abridged"]
    tags = {}

    for label in labelWords:
        if in_str.find(label) >= 0:
            tags[label] = True
            in_str = in_str.replace(label, '')

    tmp = in_str.split()
    baseTitle = ""

    index = 0
    nItems = len(tmp)
    tmp = [''.join([char for char in word if char.isalnum()]) for word in tmp]

    keyWords = ["version", "vol", "chapter", "part", "volume", "book"]
    forbiddenWords = ["a", "the", "of", "in"]

    while index < nItems:
        word = tmp[index]
        if word in keyWords and index < nItems - 1:
            if tmp[index+1].isdigit():
                tags[word] = int(tmp[index+1])
                index += 2
                continue
        elif len(word) > 0 and word not in forbiddenWords:
            if len(baseTitle) > 0:
                baseTitle += " "
            baseTitle += word
        index += 1

    return baseTitle, tags


def prepareMatches(listMetadata, pathDIR):

    authorGroups = getSameAuthorGroups(listMetadata, pathDIR)
    authorGroups = [list(authorGroups[x])
                    for x in authorGroups if len(authorGroups[x]) > 1]
    print(f"{len(authorGroups)} groups of books with the same author")
    print("Preparing the data...")

    output = []

    bar = progressbar.ProgressBar(len(authorGroups))
    bar.start()

    for index, group in enumerate(authorGroups):
        bar.update(index)
        nItems = len(group)
        match = []
        for i in range(nItems):
            pathMetadata = os.path.join(pathDIR, group[i])
            with open(pathMetadata, 'rb') as file:
                title_i = json.load(file)["title"]
            baseTitle, code = getBaseTitle(title_i)
            match.append((baseTitle, code, group[i]))
        output.append(match)
    bar.finish()

    return output


def getPossibleMatches(allGroups):

    output = []
    for group in allGroups:
        group.sort(key=lambda x: x[0])
        groupSize = len(group)
        indexStart = 0
        while indexStart < groupSize - 1:
            currMatch = []
            currTitle, currTags, currMetdataName = group[indexStart]
            for indexEnd in range(indexStart + 1, groupSize):
                nextTitle, nextTags, nextMetadataName = group[indexEnd]
                isSame = True
                if currTitle == nextTitle:
                    for tag in currTags:
                        if tag in ["version", "abridged", "dramatic reading"]:
                            continue
                        if nextTags.get(tag, None) != currTags[tag]:
                            isSame = False
                            break
                    if isSame:
                        currMatch.append(nextMetadataName)
                else:
                    break
            indexStart = indexEnd
            if len(currMatch) > 0:
                currMatch.append(currMetdataName)
                output.append(currMatch)
    return output


def get_books_duplicates(pathDIRMetadata, listMetadata):

    matches = prepareMatches(listMetadata, pathDIRMetadata)
    print("Retriveing the possible matches")
    matches = getPossibleMatches(matches)
    return matches


if __name__ == "__main__":
    from sumUp import getAllMetadata

    pathDIRMetadata = "/checkpoint/mriviere/LibriVox_full_metadata/"
    pathOut = "/checkpoint/mriviere/LibriVox_titleDuplicates.json"
    listMetadata = getAllMetadata(pathDIRMetadata)

    matches = get_books_duplicates(pathDIRMetadata, listMetadata)

    with open(pathOut, 'w') as file:
        json.dump(matches, file, indent=2)
