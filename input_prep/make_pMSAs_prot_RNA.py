#!/usr/bin/env python
import numpy as np
import string
import gzip
import os
import sys
import re

TABLE = str.maketrans(dict.fromkeys(string.ascii_lowercase))
ALPHABET = np.array(list("ARNDCQEGHILKMFPSTWYV-"), dtype='|S1').view(np.uint8)
remove_lower = lambda text: re.sub('[a-z]', '', text)
RNA_ALPHABET = np.array(list("ACGT-"), dtype='|S1').view(np.uint8)

def seq2number(seq):
    seq_no_ins = seq.translate(TABLE)
    seq_no_ins = np.array(list(seq_no_ins), dtype='|S1').view(np.uint8)
    for i in range(ALPHABET.shape[0]):
        seq_no_ins[seq_no_ins == ALPHABET[i]] = i
    seq_no_ins[seq_no_ins > 20] = 20

    return seq_no_ins

def rnaseq2number(seq):
    seq_no_ins = seq.translate(TABLE)
    seq_no_ins = np.array(list(seq_no_ins), dtype='|S1').view(np.uint8)
    for i in range(RNA_ALPHABET.shape[0]):
        seq_no_ins[seq_no_ins == RNA_ALPHABET[i]] = i
    seq_no_ins[seq_no_ins > 5] = 5

    return seq_no_ins

def calc_seqID(query, cand):
    same = (query == cand).sum()
    return same / float(len(query))

def read_a3m(fn):
    # read sequences in a3m file
    # only take one (having the highest seqID to query) per each taxID
    is_first = True
    tmp = {}
    if fn.split('.')[-1] == "gz":
        fp = gzip.open(fn, 'rt')
    else:
        fp = open(fn, 'r')

    for line in fp:
        if line[0] == ">":
            if is_first:
                continue
            x = line.split()
            seqID = x[0][1:]
            try:
                idx = line.index("TaxID")
                is_ignore = False
            except:
                is_ignore = True
                continue
            TaxID = line[idx:].split()[0].split('=')[-1]
            if not TaxID in tmp:
                tmp[TaxID] = list()
        else:
            if is_first:
                query = line.strip()
                is_first = False
            elif is_ignore:
                continue
            else:
                tmp[TaxID].append((seqID, line.strip()))

    query_in_num = seq2number(query)
    a3m = {}
    for TaxID in tmp:
        if len(tmp[TaxID]) < 1:
            continue
        if len(tmp[TaxID]) < 2:
            a3m[TaxID] = tmp[TaxID][0]
            continue
        # Get the best sequence only
        score_s = list()
        for seqID, seq in tmp[TaxID]:
            seq_in_num = seq2number(seq)
            score = calc_seqID(query_in_num, seq_in_num)
            score_s.append(score)
        #
        idx = np.argmax(score_s)
        a3m[TaxID] = tmp[TaxID][idx]

    return query, a3m

def read_afa(fn):
    # read sequences in afa file (RNA)
    # only take one (having the highest seqID to query) per each taxID
    is_first = True
    tmp = {}
    if fn.split('.')[-1] == "gz":
        fp = gzip.open(fn, 'rt')
    else:
        fp = open(fn, 'r')

    for line in fp:
        if line[0] == ">":
            if is_first:
                continue
            x = line.split()
            seqID = x[0][1:]
            try:
                idx = line.index("TaxID")
                is_ignore = False
            except:
                is_ignore = True
                continue
            TaxID = line[idx:].split('/')[0].split('=')[-1]
            if not TaxID in tmp:
                tmp[TaxID] = list()
        else:
            if is_first:
                query = line.strip()
                is_first = False
            elif is_ignore:
                continue
            else:
                tmp[TaxID].append((seqID, line.strip()))

    query_in_num = rnaseq2number(query)
    a3m = {}
    for TaxID in tmp:
        if len(tmp[TaxID]) < 1:
            continue
        if len(tmp[TaxID]) < 2:
            a3m[TaxID] = tmp[TaxID][0]
            continue
        # Get the best sequence only
        score_s = list()
        for seqID, seq in tmp[TaxID]:
            seq_in_num = rnaseq2number(seq)
            score = calc_seqID(query_in_num, seq_in_num)
            score_s.append(score)
        #
        idx = np.argmax(score_s)
        a3m[TaxID] = tmp[TaxID][idx]

    return query, a3m

def main(fnA, fnB, pair_fn):
    queryA, a3mA = read_a3m(fnA)
    queryB, a3mB = read_afa(fnB)

    #fnA_filt = fnA.split('.a3m')[0] + '.i90.c75.a3m'
    #fnB_filt = fnB.split('.a3m')[0] + '.i90.c75.a3m'
    #
    #def read_filt(filename):
    #    all_seqs = []
    #    name = ''
    #    seq = ''
    #    with open(filename) as fp:
    #        queryname = fp.readline().strip().split()[0][1:]
    #        query = fp.readline().strip()
    #        qlen = len(query)
    #        for line in fp:
    #            if line[0] == '>':
    #                lineparts = line.strip().split()
    #                if name and seq and name != queryname:
    #                    match = 0
    #                    for i in range(qlen):
    #                        if query[i] == seq[i]:
    #                            match += 1
    #                    all_seqs.append([name, seq, match])
    #                name = lineparts[0][1:]
    #                seq = ''
    #            else:
    #                seq += remove_lower(line[:-1])
    #
    #    if name and seq:
    #        match = 0
    #        for i in range(qlen):
    #            if query[i] == seq[i]:
    #                match += 1
    #        all_seqs.append([name, seq, match])
    #
    #    all_seqs.sort(key = lambda x:x[2], reverse = True)
    #    return all_seqs
    #
    #filtA = read_filt(fnA_filt)
    #filtB = read_filt(fnB_filt)

    wrt = '>query\n'
    wrt += queryA
    wrt += '/'
    wrt += queryB
    wrt += "\n"
    wrt2 = ''

    wrtlen = 0
    doneset = set([])
    for taxA in a3mA:
        if taxA in a3mB:
            wrt += ">%s %s\n"%(a3mA[taxA][0], a3mB[taxA][0])
            wrt += "%s/%s\n"%(remove_lower(a3mA[taxA][1]), remove_lower(a3mB[taxA][1]))
            wrtlen += 1
            doneset.add(a3mA[taxA][0])
            doneset.add(a3mB[taxA][0])

        elif taxA not in doneset:
            wrt2 += ">%s %s\n"%(a3mA[taxA][0], 'singlerep')
            wrt2 += "%s%s\n"%(remove_lower(a3mA[taxA][1]), '-'*len(queryB))
            wrtlen += 1
            doneset.add(a3mA[taxA][0])

    for taxB in a3mB:
        if taxB not in doneset:
            wrt2 += ">%s %s\n"%(a3mB[taxB][0], 'singlerep')
            wrt2 += "%s%s\n"%('-'*len(queryA), remove_lower(a3mB[taxB][1]))
            wrtlen += 1
            doneset.add(a3mB[taxB][0])


    with open(pair_fn, 'wt') as fp:
        fp.write(wrt)
        fp.write(wrt2)

    #    if wrtlen <= 3000:
    #        for A in filtA[:1500]:
    #            if A[0] not in doneset:
    #                fp.write('>' + A[0] + ' singleA\n' + A[1] + '/' + '-'*len(queryB) + '\n')
    #
    #        for B in filtB[:1500]:
    #            if B[0] not in doneset:
    #                fp.write('>' + B[0] + ' singleB\n' + '-'*len(queryA) + '/' + B[1] + '\n')

    print(str(wrtlen) + '\t' + pair_fn)


if __name__ == '__main__':

    if len(sys.argv) == 1:
        print ("USAGE: python make_paired_MSA_simple.py [a3m for chain A] [a3m for chain B] [output filename]")
        sys.exit()

    fnA = sys.argv[1]
    fnB = sys.argv[2]
    pair_fn = sys.argv[3]

    main(fnA, fnB, pair_fn)
