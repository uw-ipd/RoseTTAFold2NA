import torch
from torch.utils import data
import os
import csv
from dateutil import parser
import numpy as np
from scipy.sparse.csgraph import shortest_path
from parsers import parse_a3m, parse_pdb, parse_fasta_if_exists, parse_mixed_fasta
from chemical import INIT_CRDS, INIT_NA_CRDS, NAATOKENS, MASKINDEX, NTOTAL
from util import center_and_realign_missing, random_rot_trans
import pickle
import random
from os.path import exists

base_dir = "/projects/ml/TrRosetta/PDB-2021AUG02"
compl_dir = "/projects/ml/RoseTTAComplex"
#na_dir = "/projects/ml/nucleic"
na_dir = "/home/dimaio/TrRosetta/nucleic" ##!!
fb_dir = "/projects/ml/TrRosetta/fb_af"
if not os.path.exists(base_dir):
    # training on blue
    base_dir = "/gscratch2/PDB-2021AUG02"
    compl_dir = "/gscratch2/RoseTTAComplex"
    na_dir = "/gscratch2/nucleic"
    fb_dir = "/gscratch2/fb_af1"

def set_data_loader_params(args):
    PARAMS = {
        "COMPL_LIST"       : "%s/list.hetero.csv"%compl_dir,
        "HOMO_LIST"        : "%s/list.homo.csv"%compl_dir,
        "NEGATIVE_LIST"    : "%s/list.negative.csv"%compl_dir,
        "RNA_LIST"         : "%s/list.rnaonly.csv"%na_dir,
        "NA_COMPL_LIST"    : "%s/list.nucleic.v2.csv"%na_dir,
        "NEG_NA_COMPL_LIST": "%s/list.na_negatives.v2.csv"%na_dir,
        "PDB_LIST"         : "%s/list_v02.csv"%base_dir, # on digs
        #"PDB_LIST"        : "/gscratch2/list_2021AUG02.csv", # on blue
        "FB_LIST"          : "%s/list_b1-3.csv"%fb_dir,
        "VAL_PDB"          : "%s/val/xaa"%base_dir,
        "VAL_RNA"          : "%s/rna_valid.csv"%na_dir,
        "VAL_COMPL"        : "%s/val_lists/xaa"%compl_dir,
        "VAL_NEG"          : "%s/val_lists/xaa.neg"%compl_dir,
        "DATAPKL"          : "./dataset.pkl", # cache for faster loading
        "PDB_DIR"          : base_dir,
        "FB_DIR"           : fb_dir,
        "COMPL_DIR"        : compl_dir,
        "NA_DIR"           : na_dir,
        "MINTPLT"          : 0,
        "MAXTPLT"          : 5,
        "MINSEQ"           : 1,
        "MAXSEQ"           : 1024,
        "MAXLAT"           : 128, 
        "CROP"             : 256,
        "DATCUT"           : "2020-Apr-30",
        "RESCUT"           : 4.5,
        "BLOCKCUT"         : 5,
        "PLDDTCUT"         : 70.0,
        "SCCUT"            : 90.0,
        "ROWS"             : 1,
        "SEQID"            : 95.0,
        "MAXCYCLE"         : 4
    }
    for param in PARAMS:
        if hasattr(args, param.lower()):
            PARAMS[param] = getattr(args, param.lower())
    return PARAMS

def MSABlockDeletion(msa, ins, nb=5):
    '''
    Down-sample given MSA by randomly delete blocks of sequences
    Input: MSA/Insertion having shape (N, L)
    output: new MSA/Insertion with block deletion (N', L)
    '''
    N, L = msa.shape
    block_size = max(int(N*0.3), 1)
    block_start = np.random.randint(low=1, high=N, size=nb) # (nb)
    to_delete = block_start[:,None] + np.arange(block_size)[None,:]
    to_delete = np.unique(np.clip(to_delete, 1, N-1))
    #
    mask = np.ones(N, np.bool)
    mask[to_delete] = 0

    return msa[mask], ins[mask]

def cluster_sum(data, assignment, N_seq, N_res):
    # Get statistics from clustering results (clustering extra sequences with seed sequences)
    csum = torch.zeros(N_seq, N_res, data.shape[-1]).scatter_add(0, assignment.view(-1,1,1).expand(-1,N_res,data.shape[-1]), data.float())
    return csum

def MSAFeaturize(msa, ins, params, p_mask=0.15, eps=1e-6, L_s=[]):
    '''
    Input: full MSA information (after Block deletion if necessary) & full insertion information
    Output: seed MSA features & extra sequences
    
    Seed MSA features:
        - aatype of seed sequence (20 regular aa + 1 gap/unknown + 1 mask)
        - profile of clustered sequences (22)
        - insertion statistics (2)
        - N-term or C-term? (2)
    extra sequence features:
        - aatype of extra sequence (22)
        - insertion info (1)
        - N-term or C-term? (2)
    '''
    N, L = msa.shape
    
    term_info = torch.zeros((L,2), device=msa.device).float()
    if len(L_s) < 1:
        term_info[0,0] = 1.0 # flag for N-term
        term_info[-1,1] = 1.0 # flag for C-term
    else:
        start = 0
        for L_chain in L_s:
            term_info[start, 0] = 1.0 # flag for N-term
            term_info[start+L_chain-1,1] = 1.0 # flag for C-term
            start += L_chain
        
    # raw MSA profile
    raw_profile = torch.nn.functional.one_hot(msa, num_classes=NAATOKENS)
    raw_profile = raw_profile.float().mean(dim=0) 

    # Nclust sequences will be selected randomly as a seed MSA (aka latent MSA)
    # - First sequence is always query sequence
    # - the rest of sequences are selected randomly
    Nclust = min(N, params['MAXLAT'])
    
    b_seq = list()
    b_msa_clust = list()
    b_msa_seed = list()
    b_msa_extra = list()
    b_mask_pos = list()
    for i_cycle in range(params['MAXCYCLE']):
        sample = torch.randperm(N-1, device=msa.device)
        msa_clust = torch.cat((msa[:1,:], msa[1:,:][sample[:Nclust-1]]), dim=0)
        ins_clust = torch.cat((ins[:1,:], ins[1:,:][sample[:Nclust-1]]), dim=0)

        # 15% random masking 
        # - 10%: aa replaced with a uniformly sampled random amino acid
        # - 10%: aa replaced with an amino acid sampled from the MSA profile
        # - 10%: not replaced
        # - 70%: replaced with a special token ("mask")
        random_aa = torch.tensor([[0.05]*20 + [0.0]*(NAATOKENS-20)], device=msa.device)
        same_aa = torch.nn.functional.one_hot(msa_clust, num_classes=NAATOKENS)
        probs = 0.1*random_aa + 0.1*raw_profile + 0.1*same_aa
        #probs = torch.nn.functional.pad(probs, (0, 1), "constant", 0.7)
        probs[...,MASKINDEX]=0.7
        
        sampler = torch.distributions.categorical.Categorical(probs=probs)
        mask_sample = sampler.sample()

        mask_pos = torch.rand(msa_clust.shape, device=msa_clust.device) < p_mask
        mask_pos[msa_clust>MASKINDEX]=False # no masking on NAs

        msa_masked = torch.where(mask_pos, mask_sample, msa_clust)
        b_seq.append(msa_masked[0].clone())
       
        ## get extra sequenes
        if N - Nclust >= params['MAXSEQ']: # there are enough extra sequences
            Nextra = params['MAXSEQ']
            msa_extra = torch.cat((msa_masked[:1,:], msa[1:,:][sample[Nclust-1:]]), dim=0) 
            ins_extra = torch.cat((ins_clust[:1,:], ins[1:,:][sample[Nclust-1:]]), dim=0)
            extra_mask = torch.full(msa_extra.shape, False, device=msa_extra.device)
            extra_mask[0] = mask_pos[0]
        elif N - Nclust < 1: # no extra sequences, use all masked seed sequence as extra one
            Nextra = Nclust
            msa_extra = msa_masked.clone()
            ins_extra = ins_clust.clone()
            extra_mask = mask_pos.clone()
        else: # it has extra sequences, but not enough to maxseq. Use mixture of seed (except query) & extra
            Nextra = min(N, params['MAXSEQ'])
            msa_add = msa[1:,:][sample[Nclust-1:]]
            ins_add = ins[1:,:][sample[Nclust-1:]]
            mask_add = torch.full(msa_add.shape, False, device=msa_add.device)
            msa_extra = torch.cat((msa_masked, msa_add), dim=0)
            ins_extra = torch.cat((ins_clust, ins_add), dim=0)
            extra_mask = torch.cat((mask_pos, mask_add), dim=0)
        N_extra_pool = msa_extra.shape[0]
        
        # 1. one_hot encoded aatype: msa_clust_onehot
        msa_clust_onehot = torch.nn.functional.one_hot(msa_masked, num_classes=NAATOKENS) # (N, L, 22)
        msa_extra_onehot = torch.nn.functional.one_hot(msa_extra, num_classes=NAATOKENS)
        
        # clustering (assign remaining sequences to their closest cluster by Hamming distance
        count_clust = torch.logical_and(~mask_pos, msa_clust != 20) # 20: index for gap, ignore both masked & gaps
        count_extra = torch.logical_and(~extra_mask, msa_extra != 20) 
        # get number of identical tokens for each pair of sequences (extra vs seed)
        agreement = torch.matmul((count_extra[:,:,None]*msa_extra_onehot).view(N_extra_pool, -1), (count_clust[:,:,None]*msa_clust_onehot).view(Nclust, -1).T) # (N_extra_pool, Nclust)
        assignment = torch.argmax(agreement, dim=-1) # map each extra seq to the closest seed seq

        # 2. cluster profile -- ignore masked token when calculate profiles
        count_extra = ~extra_mask # only consider non-masked tokens in extra seqs
        count_clust = ~mask_pos # only consider non-masked tokens in seed seqs
        msa_clust_profile = cluster_sum(count_extra[:,:,None]*msa_extra_onehot, assignment, Nclust, L)
        msa_clust_profile += count_clust[:,:,None]*msa_clust_profile
        count_profile = cluster_sum(count_extra[:,:,None], assignment, Nclust, L).view(Nclust, L) # 
        count_profile += count_clust
        count_profile += eps
        msa_clust_profile /= count_profile[:,:,None]

        # 3. insertion statistics
        msa_clust_del = cluster_sum((count_extra*ins_extra)[:,:,None], assignment, Nclust, L).view(Nclust, L)
        msa_clust_del += count_clust*ins_clust
        msa_clust_del /= count_profile
        ins_clust = (2.0/np.pi)*torch.arctan(ins_clust.float()/3.0) # (from 0 to 1)
        msa_clust_del = (2.0/np.pi)*torch.arctan(msa_clust_del.float()/3.0) # (from 0 to 1)
        ins_clust = torch.stack((ins_clust, msa_clust_del), dim=-1)
        
        # seed MSA features (one-hot aa, cluster profile, ins statistics, terminal info)
        msa_seed = torch.cat((msa_clust_onehot, msa_clust_profile, ins_clust, term_info[None].expand(Nclust,-1,-1)), dim=-1)

        # extra MSA features (one-hot aa, insertion, terminal info)
        ins_extra = (2.0/np.pi)*torch.arctan(ins_extra[:Nextra].float()/3.0) # (from 0 to 1)
        msa_extra = torch.cat((msa_extra_onehot[:Nextra], ins_extra[:,:,None], term_info[None].expand(Nextra,-1,-1)), dim=-1)

        b_msa_clust.append(msa_clust)
        b_msa_seed.append(msa_seed)
        b_msa_extra.append(msa_extra)
        b_mask_pos.append(mask_pos)
    
    b_seq = torch.stack(b_seq)
    b_msa_clust = torch.stack(b_msa_clust)
    b_msa_seed = torch.stack(b_msa_seed)
    b_msa_extra = torch.stack(b_msa_extra)
    b_mask_pos = torch.stack(b_mask_pos)

    return b_seq, b_msa_clust, b_msa_seed, b_msa_extra, b_mask_pos

def TemplFeaturize(tplt, qlen, params, offset=0, npick=1, npick_global=None, pick_top=True, random_noise=5.0):
    if npick_global == None:
        npick_global=max(npick, 1)
    seqID_cut = params['SEQID']

    ntplt = len(tplt['ids'])
    if (ntplt < 1) or (npick < 1): #no templates in hhsearch file or not want to use templ - return fake templ
        xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(npick_global,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((npick_global, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global,qlen,NTOTAL), False)
        return xyz, t1d, mask_t

    # ignore templates having too high seqID
    if seqID_cut <= 100.0:
        tplt_valid_idx = torch.where(tplt['f0d'][0,:,4] < seqID_cut)[0]
        tplt['ids'] = np.array(tplt['ids'])[tplt_valid_idx]
    else:
        tplt_valid_idx = torch.arange(len(tplt['ids']))
    
    # check again if there are templates having seqID < cutoff
    ntplt = len(tplt['ids'])
    npick = min(npick, ntplt)
    if npick<1: # no templates -- return fake templ
        xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(npick_global,qlen,1,3)*random_noise
        t1d = torch.nn.functional.one_hot(torch.full((npick_global, qlen), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
        conf = torch.zeros((npick_global, qlen, 1)).float()
        t1d = torch.cat((t1d, conf), -1)
        mask_t = torch.full((npick_global,qlen,NTOTAL), False)
        return xyz, t1d, mask_t

    if not pick_top: # select randomly among all possible templates
        sample = torch.randperm(ntplt)[:npick]
    else: # only consider top 50 templates
        sample = torch.randperm(min(50,ntplt))[:npick]

    xyz = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(npick_global,qlen,1,1) + torch.rand(1,qlen,1,3)*random_noise
    mask_t = torch.full((npick_global,qlen,NTOTAL),False) # True for valid atom, False for missing atom
    t1d = torch.full((npick_global, qlen), 20).long()
    t1d_val = torch.zeros((npick_global, qlen)).float()

    for i,nt in enumerate(sample):
        tplt_idx = tplt_valid_idx[nt]
        sel = torch.where(tplt['qmap'][0,:,1]==tplt_idx)[0]
        pos = tplt['qmap'][0,sel,0] + offset
        xyz[i,pos,:14] = tplt['xyz'][0,sel]
        mask_t[i,pos,:14] = tplt['mask'][0,sel].bool()
        # 1-D features: alignment confidence 
        t1d[i,pos] = tplt['seq'][0,sel]
        t1d_val[i,pos] = tplt['f1d'][0,sel,2] # alignment confidence
        xyz[i] = center_and_realign_missing(xyz[i], mask_t[i])

    t1d = torch.nn.functional.one_hot(t1d, num_classes=NAATOKENS-1).float()
    t1d = torch.cat((t1d, t1d_val[...,None]), dim=-1)

    return xyz, t1d, mask_t

def get_train_valid_set(params, OFFSET=1000000):
    if (not os.path.exists(params['DATAPKL'])):
        # read validation IDs for PDB set
        val_pdb_ids = set([int(l) for l in open(params['VAL_PDB']).readlines()])
        val_compl_ids = set([int(l) for l in open(params['VAL_COMPL']).readlines()])
        val_neg_ids = set([int(l)+OFFSET for l in open(params['VAL_NEG']).readlines()])
        val_rna_pdbids = set([l.rstrip() for l in open(params['VAL_RNA']).readlines()])

        # compile NA complex sets
        # if the protein _or_ RNA is in corresponding validation set,
        #    place that example in the validation set
        with open(params['NA_COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, topad?
            rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')], bool(int(r[6]))] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        train_na_compl = {}
        valid_na_compl = {}
        for r in rows:
            prot_in_valid_set = r[2] in val_compl_ids
            rna_in_valid_set = any([x in val_rna_pdbids for x in r[0].split(':')])
            if prot_in_valid_set or rna_in_valid_set:
                if r[2] in valid_na_compl.keys():
                    valid_na_compl[r[2]].append((r[:2], r[-2], r[-1])) # ((pdb, hash), length, topad?)
                else:
                    valid_na_compl[r[2]] = [(r[:2], r[-2], r[-1])]
            else:
                if r[2] in train_na_compl.keys():
                    train_na_compl[r[2]].append((r[:2], r[-2], r[-1]))
                else:
                    train_na_compl[r[2]] = [(r[:2], r[-2], r[-1])]

        # compile negative examples
        # remove pairs if any of the subunits are included in validation set
        # cluster based on pdb cluster id
        with open(params['NEG_NA_COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
            rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')], r[6],r[7]] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        train_na_neg = {}
        valid_na_neg = {}
        for i,r in enumerate(rows):
            prot_in_valid_set = (r[2]-OFFSET) in val_compl_ids
            if prot_in_valid_set:
                valid_na_neg[i] = [(r[:2], r[-3], (r[-2],r[-1]) )]
                #if r[2] in valid_na_neg.keys():
                #    valid_na_neg[r[2]].append((r[:2], r[-3], (r[-2],r[-1]) ))
                #else:
                #    valid_na_neg[r[2]] = [(r[:2], r[-3], (r[-2],r[-1]) )]
            else:
                train_na_neg[i] = [(r[:2], r[-3], (r[-2],r[-1]) )]
                #if r[2] in train_na_neg.keys():
                #    train_na_neg[r[2]].append((r[:2], r[-3], (r[-2],r[-1]) ))
                #else:
                #    train_na_neg[r[2]] = [(r[:2], r[-3], (r[-2],r[-1]) )]

        # read & clean RNA list
        with open(params['RNA_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],[int(clid) for clid in r[3].split(':')], [int(plen) for plen in r[4].split(':')]] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        # compile training and validation sets
        train_rna = {}
        valid_rna = {}
        for i,r in enumerate(rows):
            #if any([x in val_rna_pdbids for x in r[0].split(':')]):
            #    valid_rna[i] = [(r[0], r[-1])]
            #else:
            #    train_rna[i] = [(r[0], r[-1])]

            if any([x in val_rna_pdbids for x in r[0].split(':')]):
                if r[1][0] in valid_rna.keys():
                    valid_rna[r[1][0]].append((r[0], r[-1]))
                else:
                    valid_rna[r[1][0]] = [(r[0], r[-1])]
            else:
                if r[1][0] in train_rna.keys():
                    train_rna[r[1][0]].append((r[0], r[-1]))
                else:
                    train_rna[r[1][0]] = [(r[0], r[-1])]

        # read homo-oligomer list
        homo = {}
        with open(params['HOMO_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read pdbA, pdbB, bioA, opA, bioB, opB
            rows = [[r[0], r[1], int(r[2]), int(r[3]), int(r[4]), int(r[5])] for r in reader]
        for r in rows:
            if r[0] in homo.keys():
                homo[r[0]].append(r[1:])
            else:
                homo[r[0]] = [r[1:]]

        # read & clean list.csv
        with open(params['PDB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[3],int(r[4]), int(r[-1].strip())] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        # compile training and validation sets
        val_hash = list()
        train_pdb = {}
        valid_pdb = {}
        valid_homo = {}
        for r in rows:
            if r[2] in val_pdb_ids:
                val_hash.append(r[1])
                if r[2] in valid_pdb.keys():
                    valid_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    valid_pdb[r[2]] = [(r[:2], r[-1])]
                #
                if r[0] in homo:
                    if r[2] in valid_homo.keys():
                        valid_homo[r[2]].append((r[:2], r[-1]))
                    else:
                        valid_homo[r[2]] = [(r[:2], r[-1])]
            else:
                if r[2] in train_pdb.keys():
                    train_pdb[r[2]].append((r[:2], r[-1]))
                else:
                    train_pdb[r[2]] = [(r[:2], r[-1])]

        # compile facebook model sets
        with open(params['FB_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            rows = [[r[0],r[2],int(r[3]),len(r[-1].strip())] for r in reader
                     if float(r[1]) > 80.0 and
                     len(r[-1].strip()) > 200]
        fb = {}
        for r in rows:
            if r[2] in fb.keys():
                fb[r[2]].append((r[:2], r[-1]))
            else:
                fb[r[2]] = [(r[:2], r[-1])]

        # compile complex sets
        with open(params['COMPL_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxID, assembly (bioA,opA,bioB,opB)
            rows = [[r[0], r[3], int(r[4]), [int(plen) for plen in r[5].split(':')], r[6] , [int(r[7]), int(r[8]), int(r[9]), int(r[10])]] for r in reader
                    if float(r[2]) <= params['RESCUT'] and
                    parser.parse(r[1]) <= parser.parse(params['DATCUT'])]

        train_compl = {}
        valid_compl = {}
        for r in rows:
            if r[2] in val_compl_ids:
                if r[2] in valid_compl.keys():
                    valid_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1])) # ((pdb, hash), length, taxID, assembly, negative?)
                else:
                    valid_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]
            else:
                # if subunits are included in PDB validation set, exclude them from training
                hashA, hashB = r[1].split('_')
                if hashA in val_hash:
                    continue
                if hashB in val_hash:
                    continue
                if r[2] in train_compl.keys():
                    train_compl[r[2]].append((r[:2], r[-3], r[-2], r[-1]))
                else:
                    train_compl[r[2]] = [(r[:2], r[-3], r[-2], r[-1])]

        # compile negative examples
        # remove pairs if any of the subunits are included in validation set
        with open(params['NEGATIVE_LIST'], 'r') as f:
            reader = csv.reader(f)
            next(reader)
            # read complex_pdb, pMSA_hash, complex_cluster, length, taxonomy
            rows = [[r[0],r[3],OFFSET+int(r[4]),[int(plen) for plen in r[5].split(':')],r[6]] for r in reader
                    if float(r[2])<=params['RESCUT'] and
                    parser.parse(r[1])<=parser.parse(params['DATCUT'])]

        train_neg = {}
        valid_neg = {}
        for r in rows:
            if r[2] in val_neg_ids:
                if r[2] in valid_neg.keys():
                    valid_neg[r[2]].append((r[:2], r[-2], r[-1], []))
                else:
                    valid_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]
            else:
                hashA, hashB = r[1].split('_')
                if hashA in val_hash:
                    continue
                if hashB in val_hash:
                    continue
                if r[2] in train_neg.keys():
                    train_neg[r[2]].append((r[:2], r[-2], r[-1], []))
                else:
                    train_neg[r[2]] = [(r[:2], r[-2], r[-1], [])]

        # Get average chain length in each cluster and calculate weights
        pdb_IDs = list(train_pdb.keys())
        fb_IDs = list(fb.keys())
        compl_IDs = list(train_compl.keys())
        neg_IDs = list(train_neg.keys())
        na_compl_IDs = list(train_na_compl.keys())
        na_neg_IDs = list(train_na_neg.keys())
        rna_IDs = list(train_rna.keys())

        pdb_weights = list()
        fb_weights = list()
        compl_weights = list()
        neg_weights = list()
        na_compl_weights = list()
        na_neg_weights = list()
        rna_weights = list()
        for key in pdb_IDs:
            plen = sum([plen for _, plen in train_pdb[key]]) // len(train_pdb[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            pdb_weights.append(w)
    
        for key in fb_IDs:
            plen = sum([plen for _, plen in fb[key]]) // len(fb[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            fb_weights.append(w)
    
        for key in compl_IDs:
            plen = sum([sum(plen) for _, plen, _, _ in train_compl[key]]) // len(train_compl[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            compl_weights.append(w)
    
        for key in neg_IDs:
            plen = sum([sum(plen) for _, plen, _, _ in train_neg[key]]) // len(train_neg[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            neg_weights.append(w)

        for key in na_compl_IDs:
            plen = sum([sum(plen) for _, plen, _ in train_na_compl[key]]) // len(train_na_compl[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            # 5x increase for base-specific!
            if (any([pad for _,_,pad in train_na_compl[key]])):
                w *= 5.0
            na_compl_weights.append(w)
    
        for key in na_neg_IDs:
            plen = sum([sum(plen) for _, plen, _ in train_na_neg[key]]) // len(train_na_neg[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            na_neg_weights.append(w)

        for key in rna_IDs:
            plen = sum([sum(plen) for _, plen in train_rna[key]]) // len(train_rna[key])
            w = (1/512.)*max(min(float(plen),512.),256.)
            rna_weights.append(w)

        # save
        obj = (
            pdb_IDs, pdb_weights, train_pdb,
            fb_IDs, fb_weights, fb,
            compl_IDs, compl_weights, train_compl,
            neg_IDs, neg_weights, train_neg,
            na_compl_IDs, na_compl_weights, train_na_compl,
            na_neg_IDs, na_neg_weights, train_na_neg,
            rna_IDs, rna_weights, train_rna,
            valid_pdb, valid_homo, 
            valid_compl, valid_neg,
            valid_na_compl, valid_na_neg,
            valid_rna,
            homo
        )
        with open(params["DATAPKL"], "wb") as f:
            print ('Writing',params["DATAPKL"],'...')
            pickle.dump(obj, f)
            print ('...done')

    else:

        with open(params["DATAPKL"], "rb") as f:
            print ('Loading',params["DATAPKL"],'...')
            (
                pdb_IDs, pdb_weights, train_pdb,
                fb_IDs, fb_weights, fb,
                compl_IDs, compl_weights, train_compl,
                neg_IDs, neg_weights, train_neg,
                na_compl_IDs, na_compl_weights, train_na_compl,
                na_neg_IDs, na_neg_weights, train_na_neg,
                rna_IDs, rna_weights, train_rna,
                valid_pdb, valid_homo, 
                valid_compl, valid_neg,
                valid_na_compl, valid_na_neg,
                valid_rna,
                homo
            ) = pickle.load(f)
            print ('...done')

    return (
        (pdb_IDs, torch.tensor(pdb_weights).float(), train_pdb), \
        (fb_IDs, torch.tensor(fb_weights).float(), fb), \
        (compl_IDs, torch.tensor(compl_weights).float(), train_compl), \
        (neg_IDs, torch.tensor(neg_weights).float(), train_neg),\
        (na_compl_IDs, torch.tensor(na_compl_weights).float(), train_na_compl),\
        (na_neg_IDs, torch.tensor(na_neg_weights).float(), train_na_neg),\
        (rna_IDs, torch.tensor(rna_weights).float(), train_rna),\
        valid_pdb, valid_homo, 
        valid_compl, valid_neg,
        valid_na_compl, valid_na_neg,
        valid_rna,
        homo
    )

# slice long chains
def get_crop(l, mask, device, crop_size, unclamp=False):
    sel = torch.arange(l,device=device)
    if l <= crop_size:
        return sel
    
    size = crop_size

    mask = ~(mask[:,:3].sum(dim=-1) < 3.0)
    exists = mask.nonzero()[0]

    if unclamp: # bias it toward N-term.. (follow what AF did.. but don't know why)
        x = np.random.randint(len(exists)) + 1
        res_idx = exists[torch.randperm(x)[0]].item()
    else:
        res_idx = exists[torch.randperm(len(exists))[0]].item()
    lower_bound = max(0, res_idx-size+1)
    upper_bound = min(l-size, res_idx+1)
    start = np.random.randint(lower_bound, upper_bound)
    return sel[start:start+size]

def get_complex_crop(len_s, mask, device, params):
    tot_len = sum(len_s)
    sel = torch.arange(tot_len, device=device)
    
    n_added = 0
    n_remaining = sum(len_s)
    preset = 0
    sel_s = list()
    for k in range(len(len_s)):
        n_remaining -= len_s[k]
        crop_max = min(params['CROP']-n_added, len_s[k])
        crop_min = min(len_s[k], max(1, params['CROP'] - n_added - n_remaining))
        
        if k == 0:
            crop_max = min(crop_max, params['CROP']-5)
        crop_size = np.random.randint(crop_min, crop_max+1)
        n_added += crop_size
        
        mask_chain = ~(mask[preset:preset+len_s[k],:3].sum(dim=-1) < 3.0)
        exists = mask_chain.nonzero()[0]
        res_idx = exists[torch.randperm(len(exists))[0]].item()
        lower_bound = max(0, res_idx - crop_size + 1)
        upper_bound = min(len_s[k]-crop_size, res_idx) + 1
        start = np.random.randint(lower_bound, upper_bound) + preset
        sel_s.append(sel[start:start+crop_size])
        preset += len_s[k]
    return torch.cat(sel_s)

def get_spatial_crop(xyz, mask, sel, len_s, params, label, cutoff=10.0, eps=1e-6):
    device = xyz.device
    
    # get interface residue
    cond = torch.cdist(xyz[:len_s[0],1], xyz[len_s[0]:,1]) < cutoff
    cond = torch.logical_and(cond, mask[:len_s[0],None,1]*mask[None,len_s[0]:,1]) 
    i,j = torch.where(cond)
    ifaces = torch.cat([i,j+len_s[0]])
    if len(ifaces) < 1:
        print ("ERROR: no iface residue????", label)
        return get_complex_crop(len_s, mask, device, params)
    cnt_idx = ifaces[np.random.randint(len(ifaces))]

    dist = torch.cdist(xyz[:,1], xyz[cnt_idx,1][None]).reshape(-1) + torch.arange(len(xyz), device=xyz.device)*eps
    cond = mask[:,1]*mask[cnt_idx,1]
    dist[~cond] = 999999.9
    _, idx = torch.topk(dist, params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])
    return sel

# this is a bit of a mess...
def get_na_crop(seq, xyz, mask, sel, len_s, params, negative=False, incl_protein=True, cutoff=12.0, bp_cutoff=4.0, eps=1e-6):
    device = xyz.device

    # get base pairing NA bases
    repatom = torch.zeros(sum(len_s), dtype=torch.long, device=xyz.device)
    repatom[seq==22] = 15 # DA - N1
    repatom[seq==23] = 14 # DC - N3
    repatom[seq==24] = 15 # DG - N1
    repatom[seq==25] = 14 # DT - N3
    repatom[seq==27] = 12 # A - N1
    repatom[seq==28] = 15 # C - N3
    repatom[seq==29] = 12 # G - N1
    repatom[seq==30] = 15 # U - N3

    if not incl_protein: # either 1 or 2 NA chains
        if len(len_s)==2:
            # 2 RNA chains
            xyz_na1_rep = torch.gather(xyz[:len_s[0]], 1, repatom[:len_s[0],None,None].repeat(1,1,3)).squeeze(1)
            xyz_na2_rep = torch.gather(xyz[len_s[0]:], 1, repatom[len_s[0]:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(mask[:len_s[0]], 1, repatom[:len_s[0],None]).squeeze(1)
            mask_na2_rep = torch.gather(mask[len_s[0]:], 1, repatom[len_s[0]:,None]).squeeze(1)
            cond = torch.logical_and(cond, mask_na1_rep[:,None]*mask_na2_rep[None,:]) 
        else:
            # 1 RNA chains
            xyz_na_rep = torch.gather(xyz, 1, repatom[:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz_na_rep, xyz_na_rep) < bp_cutoff
            mask_na_rep = torch.gather(mask, 1, repatom[:,None]).squeeze(1)
            cond = torch.logical_and(cond, mask_na_rep[:,None]*mask_na_rep[None,:])

        if (torch.sum(cond)==0):
            i= np.random.randint(len_s[0]-1)
            while (not mask[i,1] or not mask[i+1,1]):
                i = np.random.randint(len_s[0])
            cond[i,i+1] = True

    else: # either 1prot+1NA, 1prot+2NA or 2prot+2NA
        # find NA:NA basepairs
        if len(len_s)>=3:
            if len(len_s)==3:
                na1s, na2s = len_s[0], len_s[0]+len_s[1]
            else:
                na1s, na2s = len_s[0]+len_s[1], len_s[0]+len_s[1]+len_s[2]

            xyz_na1_rep = torch.gather(xyz[na1s:na2s], 1, repatom[na1s:na2s,None,None].repeat(1,1,3)).squeeze(1)
            xyz_na2_rep = torch.gather(xyz[na2s:], 1, repatom[na2s:,None,None].repeat(1,1,3)).squeeze(1)
            cond_bp = torch.cdist(xyz_na1_rep, xyz_na2_rep) < bp_cutoff

            mask_na1_rep = torch.gather(mask[na1s:na2s], 1, repatom[na1s:na2s,None]).squeeze(1)
            mask_na2_rep = torch.gather(mask[na2s:], 1, repatom[na2s:,None]).squeeze(1)
            cond_bp = torch.logical_and(cond_bp, mask_na1_rep[:,None]*mask_na2_rep[None,:])

        # find NA:prot contacts
        if (not negative):
            # get interface residues
            #   interface defined as chain 1 versus all other chains
            if len(len_s)==4:
                first_na = len_s[0]+len_s[1]
            else:
                first_na = len_s[0]

            xyz_na_rep = torch.gather(xyz[first_na:], 1, repatom[first_na:,None,None].repeat(1,1,3)).squeeze(1)
            cond = torch.cdist(xyz[:first_na,1], xyz_na_rep) < cutoff
            mask_na_rep = torch.gather(mask[first_na:], 1, repatom[first_na:,None]).squeeze(1)
            cond = torch.logical_and(
                cond, 
                mask[:first_na,None,1] * mask_na_rep[None,:]
            )

        # random NA:prot contact for negatives
        if (negative or torch.sum(cond)==0):
            if len(len_s)==4:
                nprot,nna = len_s[0]+len_s[1], sum(len_s[2:])
            else:
                nprot,nna = len_s[0], sum(len_s[1:])

            # pick a random pair of residues
            cond = torch.zeros( (nprot, nna), dtype=torch.bool )
            i,j = np.random.randint(nprot), np.random.randint(nna)
            while (not mask[i,1]):
                i = np.random.randint(nprot)
            while (not mask[nprot+j,1]):
                j = np.random.randint(nna)
            cond[i,j] = True

    # a) build a graph of costs:
    #     cost (i,j in same chain) = abs(i-j)
    #     cost (i,j in different chains) = { 0 if i,j are an interface
    #                                    = { 999 if i,j are NOT an interface
    if len(len_s)>=3:
        if len(len_s)==4:
            nprot,nna1,nna2 = len_s[0]+len_s[1], len_s[2], len_s[3]
            diag_1 = np.full((nprot,nprot),999)
            diag_1[:len_s[0],:len_s[0]] = np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:])
            diag_1[len_s[0]:,len_s[0]:] = np.abs(np.arange(len_s[1])[:,None]-np.arange(len_s[1])[None,:])
        else:
            nprot,nna1,nna2 = len_s[0], len_s[1], len_s[2]
            diag_1 = np.abs(np.arange(nprot)[:,None]-np.arange(nprot)[None,:])

        diag_2 = np.abs(np.arange(len_s[-2])[:,None]-np.arange(len_s[-2])[None,:])
        diag_3 = np.abs(np.arange(len_s[-1])[:,None]-np.arange(len_s[-1])[None,:])
        int_1_2 = np.full((nprot,nna1),999)
        int_1_3 = np.full((nprot,nna2),999)
        int_2_3 = np.full((nna1,nna2),999)
        int_1_2[cond[:,:nna1]]=1
        int_1_3[cond[:,nna1:]]=1
        int_2_3[cond_bp] = 0

        inter = np.block([
            [diag_1   , int_1_2  , int_1_3],
            [int_1_2.T, diag_2   , int_2_3],
            [int_1_3.T, int_2_3.T, diag_3]
        ])
    elif len(len_s)==2:
        int_1_2 = np.full((len_s[0],len_s[1]),999)
        int_1_2[cond]=1
        inter = np.block([
            [np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:]),int_1_2],
            [int_1_2.T,np.abs(np.arange(len_s[1])[:,None]-np.arange(len_s[1])[None,:])]
        ])
    else:
        inter = np.abs(np.arange(len_s[0])[:,None]-np.arange(len_s[0])[None,:])
        inter[cond] = 1

    # b) pick a random interface residue
    intface,_ = torch.where(cond)
    startres = intface[np.random.randint(len(intface))]

    # c) traverse graph starting from chosen residue
    d_res = shortest_path(inter,directed=False,indices=startres)
    _, idx = torch.topk(torch.from_numpy(d_res).to(device=device), params['CROP'], largest=False)

    sel, _ = torch.sort(sel[idx])

    return sel


# merge msa & insertion statistics of two proteins having different taxID
def merge_a3m_hetero(a3mA, a3mB, L_s):
    # merge msa
    query = torch.cat([a3mA['msa'][0], a3mB['msa'][0]]).unsqueeze(0) # (1, L)
    msa = [query]
    if a3mA['msa'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['msa'][1:], (0,sum(L_s[1:])), "constant", 20) # pad gaps
        msa.append(extra_A)
    if a3mB['msa'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['msa'][1:], (L_s[0],0), "constant", 20)
        msa.append(extra_B)
    msa = torch.cat(msa, dim=0)

    # merge ins
    query = torch.cat([a3mA['ins'][0], a3mB['ins'][0]]).unsqueeze(0) # (1, L)
    ins = [query]
    if a3mA['ins'].shape[0] > 1:
        extra_A = torch.nn.functional.pad(a3mA['ins'][1:], (0,sum(L_s[1:])), "constant", 0) # pad gaps
        ins.append(extra_A)
    if a3mB['ins'].shape[0] > 1:
        extra_B = torch.nn.functional.pad(a3mB['ins'][1:], (L_s[0],0), "constant", 0)
        ins.append(extra_B)
    ins = torch.cat(ins, dim=0)
    return {'msa': msa, 'ins': ins}

# merge msa & insertion statistics of units in homo-oligomers
def merge_a3m_homo(msa_orig, ins_orig, nmer):
    N, L = msa_orig.shape[:2]
    msa = torch.cat([msa_orig for imer in range(nmer)], dim=1)
    ins = torch.cat([ins_orig for imer in range(nmer)], dim=1)
    return msa, ins

# Generate input features for single-chain
def featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=False, pick_top=True, random_noise=5.0):
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    xyz_t,f1d_t,mask_t = TemplFeaturize(tplt, msa.shape[1], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
    
    # get ground-truth structures
    idx = torch.arange(len(pdb['xyz'])) 
    xyz = torch.full( (len(idx), NTOTAL, 3), np.nan )
    xyz[:,:14,:] = pdb['xyz']
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask']
    xyz = torch.nan_to_num(xyz)

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False

# Generate input features for homo-oligomers
def featurize_homo(msa_orig, ins_orig, tplt, pdbA, pdbid, interfaces, params, pick_top=True, random_noise=5.0):
    L = msa_orig.shape[1]
    
    msa, ins = merge_a3m_homo(msa_orig, ins_orig, 2) # make unpaired alignments, for training, we always use two chains
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=[L,L])

    # get template features
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    if ntempl < 1:
        xyz_t, f1d_t, mask_t = TemplFeaturize(tplt, 2*L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        xyz_prev = xyz_t[0].clone()
        mask_prev = mask_t[0].clone()
    else:
        xyz_t_single, f1d_t_single, mask_t_single = TemplFeaturize(tplt, L, params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        # duplicate
        xyz_t = torch.cat((xyz_t_single, random_rot_trans(xyz_t_single)), dim=1) # (ntempl, 2*L, natm, 3)
        f1d_t = torch.cat((f1d_t_single, f1d_t_single), dim=1) # (ntempl, 2*L, 21)
        mask_t = torch.cat((mask_t_single, mask_t_single), dim=1) # (ntempl, 2*L, natm)
    
        # get initial coordinates
        xyz_prev = xyz_t[0].clone()
        mask_prev = mask_t[0].clone()

    # get ground-truth structures
    # load metadata
    PREFIX = "%s/torch/pdb/%s/%s"%(params['PDB_DIR'],pdbid[1:3],pdbid)
    meta = torch.load(PREFIX+".pt")

    # get all possible pairs
    npairs = len(interfaces)
    xyz = torch.full( (npairs, 2*L, NTOTAL, 3), np.nan )
    mask = torch.full((npairs, 2*L, NTOTAL), False)
    for i_int,interface in enumerate(interfaces):
        pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+interface[0][1:3]+'/'+interface[0]+'.pt')
        xformA = meta['asmb_xform%d'%interface[1]][interface[2]]
        xformB = meta['asmb_xform%d'%interface[3]][interface[4]]
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        xyz[i_int,:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask[i_int,:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(L*2)
    idx[L:] += 100 # to let network know about chain breaks

    # indicator for which residues are in same chain
    chain_idx = torch.zeros((2*L, 2*L)).long()
    chain_idx[:L, :L] = 1
    chain_idx[L:, L:] = 1
    
    # Residue cropping
    if 2*L > params['CROP']:
        if np.random.rand() < 0.5: # 50% --> interface crop
            spatial_crop_tgt = np.random.randint(0, npairs)
            crop_idx = get_spatial_crop(xyz[spatial_crop_tgt], mask[spatial_crop_tgt], torch.arange(L*2), [L,L], params, interfaces[spatial_crop_tgt][0])
        else: # 50% --> have same cropped regions across all copies
            crop_idx = get_crop(L, mask[0,:L], msa_seed_orig.device, params['CROP']//2, unclamp=False) # cropped region for first copy
            crop_idx = torch.cat((crop_idx, crop_idx+L)) # get same crops
        seq = seq[:,crop_idx]
        msa_seed_orig = msa_seed_orig[:,:,crop_idx]
        msa_seed = msa_seed[:,:,crop_idx]
        msa_extra = msa_extra[:,:,crop_idx]
        mask_msa = mask_msa[:,:,crop_idx]
        xyz_t = xyz_t[:,crop_idx]
        f1d_t = f1d_t[:,crop_idx]
        mask_t = mask_t[:,crop_idx]
        xyz = xyz[:,crop_idx]
        mask = mask[:,crop_idx]
        idx = idx[crop_idx]
        chain_idx = chain_idx[crop_idx][:,crop_idx]
        xyz_prev = xyz_prev[crop_idx]
        mask_prev = mask_prev[crop_idx]
    
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False

def get_pdb(pdbfilename, plddtfilename, item, lddtcut, sccut):
    xyz, mask, res_idx = parse_pdb(pdbfilename)
    plddt = np.load(plddtfilename)
    
    # update mask info with plddt (ignore sidechains if plddt < 90.0)
    #mask_lddt = np.full_like(mask, False)
    #mask_lddt[plddt > sccut] = True
    #mask_lddt[:,:5] = True
    #mask = np.logical_and(mask, mask_lddt)
    mask = np.logical_and(mask, (plddt > lddtcut)[:,None])
    
    return {'xyz':torch.tensor(xyz), 'mask':torch.tensor(mask), 'idx': torch.tensor(res_idx), 'label':item}

def get_msa(a3mfilename, item, maxseq=10000):
    msa,ins = parse_a3m(a3mfilename, maxseq=maxseq)
    return {'msa':torch.tensor(msa), 'ins':torch.tensor(ins), 'label':item}

# Load PDB examples
def loader_pdb(item, params, homo, unclamp=False, pick_top=True, p_homo_cut=0.5):
    #print ('start loader_pdb',item)

    # load MSA, PDB, template info
    pdb = torch.load(params['PDB_DIR']+'/torch/pdb/'+item[0][1:3]+'/'+item[0]+'.pt')
    a3m = get_msa(params['PDB_DIR'] + '/a3m/' + item[1][:3] + '/' + item[1] + '.a3m.gz', item[1])
    tplt = torch.load(params['PDB_DIR']+'/torch/hhr/'+item[1][:3]+'/'+item[1]+'.pt')

    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)

    if item[0] in homo: # Target is homo-oligomer
        p_homo = np.random.rand()
        if p_homo < p_homo_cut: # model as homo-oligomer with p_homo_cut prob
            pdbid = item[0].split('_')[0]
            interfaces = homo[item[0]]
            fh = featurize_homo(msa, ins, tplt, pdb, pdbid, interfaces, params, pick_top=pick_top)
            #print ('done loader_pdb',item)
            return fh
        else:
            fsc = featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
            #print ('done loader_pdb',item)
            return fsc
    else:
        fsc = featurize_single_chain(msa, ins, tplt, pdb, params, unclamp=unclamp, pick_top=pick_top)
        #print ('done loader_pdb',item)
        return fsc

def loader_fb(item, params, unclamp=False, random_noise=5.0):
    #print ('start loader_fb',item)

    # loads sequence/structure/plddt information 
    a3m = get_msa(os.path.join(params["FB_DIR"], "a3m", item[-1][:2], item[-1][2:], item[0]+".a3m.gz"), item[0])
    pdb = get_pdb(os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".pdb"),
                  os.path.join(params["FB_DIR"], "pdb", item[-1][:2], item[-1][2:], item[0]+".plddt.npy"),
                  item[0], params['PLDDTCUT'], params['SCCUT'])
    
    # get msa features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    l_orig = msa.shape[1]
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params)
    
    # get template features -- None
    xyz_t = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,l_orig,1,1) + torch.rand(1,l_orig,1,3)*random_noise
    f1d_t = torch.nn.functional.one_hot(torch.full((1, l_orig), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    conf = torch.zeros((1, l_orig, 1)).float()
    f1d_t = torch.cat((f1d_t, conf), -1)
    mask_t = torch.full((1,l_orig,NTOTAL), False)
    
    idx = pdb['idx']
    xyz = torch.full((len(idx), NTOTAL, 3), np.nan)
    xyz[:,:14,:] = pdb['xyz'][:,:14,:]
    mask = torch.full((len(idx), NTOTAL), False)
    mask[:,:14] = pdb['mask'][:,:14]
    xyz = torch.nan_to_num(xyz)

    # Residue cropping
    crop_idx = get_crop(len(idx), mask, msa_seed_orig.device, params['CROP'], unclamp=unclamp)
    seq = seq[:,crop_idx]
    msa_seed_orig = msa_seed_orig[:,:,crop_idx]
    msa_seed = msa_seed[:,:,crop_idx]
    msa_extra = msa_extra[:,:,crop_idx]
    mask_msa = mask_msa[:,:,crop_idx]
    xyz_t = xyz_t[:,crop_idx]
    f1d_t = f1d_t[:,crop_idx]
    mask_t = mask_t[:,crop_idx]
    xyz = xyz[crop_idx]
    mask = mask[crop_idx]
    idx = idx[crop_idx]

    # initial structure
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()
    chain_idx = torch.ones((len(crop_idx), len(crop_idx))).long()
    
    #print ('done loader_fb',item)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa, \
           xyz.float(), mask, idx.long(),\
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, unclamp, False

def loader_complex(item, L_s, taxID, assem, params, negative=False, pick_top=True, random_noise=5.0):
    #print ('start loader_complex',item)

    pdb_pair = item[0]
    pMSA_hash = item[1]
    
    msaA_id, msaB_id = pMSA_hash.split('_')
    if len(set(taxID.split(':'))) == 1: # two proteins have same taxID -- use paired MSA
        # read pMSA
        if negative:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA.negative/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        else:
            pMSA_fn = params['COMPL_DIR'] + '/pMSA/' + msaA_id[:3] + '/' + msaB_id[:3] + '/' + pMSA_hash + '.a3m.gz'
        a3m = get_msa(pMSA_fn, pMSA_hash)
    else:
        # read MSA for each subunit & merge them
        a3mA_fn = params['PDB_DIR'] + '/a3m/' + msaA_id[:3] + '/' + msaA_id + '.a3m.gz'
        a3mB_fn = params['PDB_DIR'] + '/a3m/' + msaB_id[:3] + '/' + msaB_id + '.a3m.gz'
        a3mA = get_msa(a3mA_fn, msaA_id)
        a3mB = get_msa(a3mB_fn, msaB_id)
        a3m = merge_a3m_hetero(a3mA, a3mB, L_s)

    # get MSA features
    msa = a3m['msa'].long()
    if negative: # Qian's paired MSA for true-pairs have no insertions... (ignore insertion to avoid any weird bias..) 
        ins = torch.zeros_like(msa)
    else:
        ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=L_s)

    # read template info
    tpltA_fn = params['PDB_DIR'] + '/torch/hhr/' + msaA_id[:3] + '/' + msaA_id + '.pt'
    tpltB_fn = params['PDB_DIR'] + '/torch/hhr/' + msaB_id[:3] + '/' + msaB_id + '.pt'
    tpltA = torch.load(tpltA_fn)
    tpltB = torch.load(tpltB_fn)

    ntemplA = np.random.randint(params['MINTPLT'], params['MAXTPLT']+1)
    ntemplB = np.random.randint(0, params['MAXTPLT']+1-ntemplA)
    xyz_t_A, f1d_t_A, mask_t_A = TemplFeaturize(tpltA, L_s[0], params, offset=0, npick=ntemplA, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t_B, f1d_t_B, mask_t_B = TemplFeaturize(tpltB, L_s[1], params, offset=0, npick=ntemplB, npick_global=max(1,max(ntemplA, ntemplB)), pick_top=pick_top, random_noise=random_noise)
    xyz_t = torch.cat((xyz_t_A, random_rot_trans(xyz_t_B)), dim=1) # (T, L1+L2, natm, 3)
    f1d_t = torch.cat((f1d_t_A, f1d_t_B), dim=1) # (T, L1+L2, natm, 3)
    mask_t = torch.cat((mask_t_A, mask_t_B), dim=1) # (T, L1+L2, natm, 3)

    # get initial coordinates
    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    # read PDB
    pdbA_id, pdbB_id = pdb_pair.split(':')
    pdbA = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbA_id[1:3]+'/'+pdbA_id+'.pt')
    pdbB = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbB_id[1:3]+'/'+pdbB_id+'.pt')
    
    if len(assem) > 0:
        # read metadata
        pdbid = pdbA_id.split('_')[0]
        meta = torch.load(params['PDB_DIR']+'/torch/pdb/'+pdbid[1:3]+'/'+pdbid+'.pt')

        # get transform
        xformA = meta['asmb_xform%d'%assem[0]][assem[1]]
        xformB = meta['asmb_xform%d'%assem[2]][assem[3]]
        
        # apply transform
        xyzA = torch.einsum('ij,raj->rai', xformA[:3,:3], pdbA['xyz']) + xformA[:3,3][None,None,:]
        xyzB = torch.einsum('ij,raj->rai', xformB[:3,:3], pdbB['xyz']) + xformB[:3,3][None,None,:]
        #xyz = INIT_CRDS.reshape(1, NTOTAL, 3).repeat(sum(L_s), 1, 1)
        xyz = torch.full((sum(L_s), NTOTAL,3), np.nan)
        xyz[:,:14] = torch.cat((xyzA, xyzB), dim=0)
        mask = torch.full((sum(L_s), NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    else:
        #xyz = INIT_CRDS.reshape(1, NTOTAL, 3).repeat(sum(L_s), 1, 1)
        xyz = torch.full((sum(L_s), NTOTAL,3), np.nan)
        xyz[:,:14] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=0)
        mask = torch.full((sum(L_s), NTOTAL), False)
        mask[:,:14] = torch.cat((pdbA['mask'], pdbB['mask']), dim=0)
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(sum(L_s))
    idx[L_s[0]:] += 100

    chain_idx = torch.zeros((sum(L_s), sum(L_s))).long()
    chain_idx[:L_s[0], :L_s[0]] = 1
    chain_idx[L_s[0]:, L_s[0]:] = 1

    # Do cropping
    if sum(L_s) > params['CROP']:
        if negative:
            sel = get_complex_crop(L_s, mask, seq.device, params)
        else:
            sel = get_spatial_crop(xyz, mask, torch.arange(sum(L_s)), L_s, params, pdb_pair)
        #
        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[sel]
        mask = mask[sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        xyz_prev = xyz_prev[sel]
        mask_prev = mask_prev[sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
    
    #print ('done loader_complex',item)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t,\
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative

# 'padding' is different depending on 'negative'
#    if FALSE, it is a boolean that says whether or not the sequence should be padded
#    if TRUE, it is new DNA sequences that should replace the baseline
def loader_na_complex(item, Ls, padding, params, native_NA_frac=0.05, negative=False, pick_top=True, random_noise=5.0):
    pdb_set = item[0]
    msa_id = item[1]

    # read PDBs
    pdb_ids = pdb_set.split(':')

    # protein + NA
    NMDLS = 1
    if (len(pdb_ids)==2):
        pdbA = [ torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt') ]
        pdbB = [ torch.load(params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt') ]

        msaB,insB = None,None

        msaB,insB = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}

    # protein + NA duplex
    elif (len(pdb_ids)==3):
        pdbA = [ torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt') ]
        pdbB = [ 
            torch.load(params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt'),
            torch.load(params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.pt')
        ]
        msaB1,insB1 = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        msaB2,insB2 = parse_fasta_if_exists(
            pdbB[1]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        if (pdbB[0]['seq']==pdbB[1]['seq']):
            NMDLS=2 # flip B0 and B1

        a3mB1 = {'msa':torch.from_numpy(msaB1), 'ins':torch.from_numpy(insB1)}
        a3mB2 = {'msa':torch.from_numpy(msaB2), 'ins':torch.from_numpy(insB2)}
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[1:])

    # homodimer + NA duplex
    elif (len(pdb_ids)==4):
        pdbA = [
            torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt'),
            torch.load(params['PDB_DIR']+'/torch/pdb/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt')
        ]
        pdbB = [ 
            torch.load(params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.pt'),
            torch.load(params['NA_DIR']+'/torch/'+pdb_ids[3][1:3]+'/'+pdb_ids[3]+'.pt')
        ]

        msaB1,insB1 = parse_fasta_if_exists(
            pdbB[0]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[2][1:3]+'/'+pdb_ids[2]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        msaB2,insB2 = parse_fasta_if_exists(
            pdbB[1]['seq'], params['NA_DIR']+'/torch/'+pdb_ids[3][1:3]+'/'+pdb_ids[3]+'.afa', 
            maxseq=5000,
            rmsa_alphabet=True
        )
        a3mB1 = {'msa':torch.from_numpy(msaB1), 'ins':torch.from_numpy(insB1)}
        a3mB2 = {'msa':torch.from_numpy(msaB2), 'ins':torch.from_numpy(insB2)}
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[2:])

        NMDLS=2 # flip A0 and A1
        if (pdbB[0]['seq']==pdbB[1]['seq']):
            NMDLS=4 # flip B0 and B1

    else:
        assert False

    # apply padding!
    if (not negative and padding==True):
        assert (len(pdbB)==2)
        lpad = np.random.randint(6)
        rpad = np.random.randint(6)
        lseq1 = torch.randint(4,(1,lpad))
        rseq1 = torch.randint(4,(1,rpad))
        lseq2 = 3-torch.flip(rseq1,(1,))
        rseq2 = 3-torch.flip(lseq1,(1,))

        # pad seqs -- hacky, DNA indices 22-25
        msaB1 = torch.cat((22+lseq1,a3mB1['msa'],22+rseq1), dim=1)
        msaB2 = torch.cat((22+lseq2,a3mB2['msa'],22+rseq2), dim=1)
        insB1 = torch.cat((torch.zeros_like(lseq1),a3mB1['ins'],torch.zeros_like(rseq1)), dim=1)
        insB2 = torch.cat((torch.zeros_like(lseq2),a3mB2['ins'],torch.zeros_like(rseq2)), dim=1)
        a3mB1 = {'msa':msaB1, 'ins':insB1}
        a3mB2 = {'msa':msaB2, 'ins':insB2}

        # update lengths
        Ls = Ls.copy()
        Ls[-2] = msaB1.shape[1]
        Ls[-1] = msaB2.shape[1]

        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[-2:])

        # pad PDB
        pdbB[0]['xyz'] = torch.nn.functional.pad(pdbB[0]['xyz'], (0,0,0,0,lpad,rpad), "constant", 0.0)
        pdbB[0]['mask'] = torch.nn.functional.pad(pdbB[0]['mask'], (0,0,lpad,rpad), "constant", False)
        pdbB[1]['xyz'] = torch.nn.functional.pad(pdbB[1]['xyz'], (0,0,0,0,rpad,lpad), "constant", 0.0)
        pdbB[1]['mask'] = torch.nn.functional.pad(pdbB[1]['mask'], (0,0,rpad,lpad), "constant", False)

    # rewrite seq if negative!
    if (negative):
        alphabet = np.array(list("ARNDCQEGHILKMFPSTWYV-Xacgtxbdhuy"), dtype='|S1').view(np.uint8)
        seqA = np.array( [list(padding[0])], dtype='|S1').view(np.uint8)
        seqB = np.array( [list(padding[1])], dtype='|S1').view(np.uint8)
        for i in range(alphabet.shape[0]):
            seqA[seqA == alphabet[i]] = i
            seqB[seqB == alphabet[i]] = i
        seqA = torch.tensor(seqA)
        seqB = torch.tensor(seqB)

        # scramble seq
        diff = (a3mB1['msa'] != seqA)
        shift = torch.randint(1,4, (torch.sum(diff),), dtype=torch.uint8)
        seqA[diff] = ((a3mB1['msa'][diff]-22)+shift)%4+22
        seqB = torch.flip(25-seqA+22, dims=(-1,))

        #print (pdb_ids)
        #print ('o1',a3mB1['msa'].numpy())
        #print ('o2',a3mB2['msa'].numpy())
        #print ('n1',seqA.numpy())
        #print ('n2',seqB.numpy())

        a3mB1 = {'msa':seqA, 'ins':torch.zeros(seqA.shape)}
        a3mB2 = {'msa':seqB, 'ins':torch.zeros(seqB.shape)}
        a3mB = merge_a3m_hetero(a3mB1, a3mB2, Ls[-2:])

        # ???
        #Ls[-2] = seqA.shape[-1]
        #Ls[-1] = seqB.shape[-1]


    ## look for shared MSA
    a3m=None
    NAchn = pdb_ids[1].split('_')[1]
    sharedMSA = params['NA_DIR']+'/msas/'+pdb_ids[0][1:3]+'/'+pdb_ids[0][:4]+'/'+pdb_ids[0]+'_'+NAchn+'_paired.a3m'
    if (len(pdb_ids)==2 and exists(sharedMSA)):
        msa,ins,_ = parse_mixed_fasta(sharedMSA)
        if (msa.shape[1] != sum(Ls)):
            print ("Error shared MSA",pdb_ids, msa.shape, Ls)
        else:
            a3m = {'msa':torch.from_numpy(msa),'ins':torch.from_numpy(ins)}

    if (a3m is None):
        # read MSA for protein
        a3mA = get_msa(params['PDB_DIR'] + '/a3m/' + msa_id[:3] + '/' + msa_id + '.a3m.gz', msa_id, maxseq=5000)
        if (len(pdbA)==2):
            msa = a3mA['msa'].long()
            ins = a3mA['ins'].long()
            msa,ins = merge_a3m_homo(msa, ins, 2)
            a3mA = {'msa':msa,'ins':ins}

        if (len(pdb_ids)==4):
            a3m = merge_a3m_hetero(a3mA, a3mB, [Ls[0]+Ls[1],sum(Ls[2:])])
        else:
            a3m = merge_a3m_hetero(a3mA, a3mB, [Ls[0],sum(Ls[1:])])

    # the block below is due to differences in the way RNA and DNA structures are processed
    # to support NMR, RNA structs return multiple states
    # For protein/NA complexes get rid of the 'NMODEL' dimension (if present)
    # NOTE there are a very small number of protein/NA NMR models:
    #       - ideally these should return the ensemble, but that requires reprocessing of proteins
    for pdb in pdbB:
        if (len(pdb['xyz'].shape) > 3):
             pdb['xyz'] = pdb['xyz'][0,...]
             pdb['mask'] = pdb['mask'][0,...]

    # read template info
    tpltA = torch.load(params['PDB_DIR'] + '/torch/hhr/' + msa_id[:3] + '/' + msa_id + '.pt')
    ntempl = np.random.randint(params['MINTPLT'], params['MAXTPLT']-1)
    if (len(pdb_ids)==4):
        if ntempl < 1:
            xyz_t, f1d_t, mask_t = TemplFeaturize(tpltA, 2*Ls[0], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
        else:
            xyz_t_single, f1d_t_single, mask_t_single = TemplFeaturize(tpltA, Ls[0], params, npick=ntempl, offset=0, pick_top=pick_top, random_noise=random_noise)
            # duplicate
            xyz_t = torch.cat((xyz_t_single, random_rot_trans(xyz_t_single)), dim=1) # (ntempl, 2*L, natm, 3)
            f1d_t = torch.cat((f1d_t_single, f1d_t_single), dim=1) # (ntempl, 2*L, 21)
            mask_t = torch.cat((mask_t_single, mask_t_single), dim=1) # (ntempl, 2*L, natm)

        ntmpl = xyz_t.shape[0]
        nNA = sum(Ls[2:])
        xyz_t = torch.cat( 
            (xyz_t, INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(ntmpl,nNA,1,1) + torch.rand(ntmpl,nNA,1,3)*random_noise), dim=1)
        f1d_t = torch.cat(
            (f1d_t, torch.nn.functional.one_hot(torch.full((ntmpl,nNA), 20).long(), num_classes=NAATOKENS).float()), dim=1) # add extra class for 0 confidence
        mask_t = torch.cat( 
            (mask_t, torch.full((ntmpl,nNA,NTOTAL), False)), dim=1)

        NAstart = 2*Ls[0]
    else:
        xyz_t, f1d_t, mask_t = TemplFeaturize(tpltA, sum(Ls), params, offset=0, npick=ntempl, pick_top=pick_top, random_noise=random_noise)
        xyz_t[:,Ls[0]:] = INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls[1:]),1,1) + torch.rand(1,sum(Ls[1:]),1,3)*random_noise
        NAstart = Ls[0]

    # seed with native NA
    if (np.random.rand()<=native_NA_frac):
        natNA_templ = torch.cat( [x['xyz'] for x in pdbB], dim=0)
        maskNA_templ = torch.cat( [x['mask'] for x in pdbB], dim=0)

        # construct template from NA
        xyz_t_B = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(1,sum(Ls),1,1) + torch.rand(1,sum(Ls),1,3)*random_noise
        mask_t_B = torch.full((1,sum(Ls),NTOTAL), False)
        mask_t_B[:,NAstart:,:23] = maskNA_templ
        xyz_t_B[mask_t_B] = natNA_templ[maskNA_templ]

        seq_t_B = torch.cat( (torch.full((1, NAstart), 20).long(),  a3mB['msa'][0:1]), dim=1)
        seq_t_B[seq_t_B>21] -= 1 # remove mask token
        f1d_t_B = torch.nn.functional.one_hot(seq_t_B, num_classes=NAATOKENS-1).float()
        conf_B = torch.cat( (
            torch.zeros((1,NAstart,1)),
            torch.full((1,sum(Ls)-NAstart,1),1.0),
        ),dim=1).float()
        f1d_t_B = torch.cat((f1d_t_B, conf_B), -1)

        xyz_t = torch.cat((xyz_t,xyz_t_B),dim=0)
        f1d_t = torch.cat((f1d_t,f1d_t_B),dim=0)
        mask_t = torch.cat((mask_t,mask_t_B),dim=0)

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    if len(msa) > params['BLOCKCUT']:
        msa, ins = MSABlockDeletion(msa, ins)
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    # build native from components
    xyz = torch.full((NMDLS, sum(Ls), NTOTAL, 3), np.nan)
    mask = torch.full((NMDLS, sum(Ls), NTOTAL), False)
    if (len(pdb_ids)==2):
        xyz[0,:NAstart,:14] = pdbA[0]['xyz']
        xyz[0,NAstart:,:23] = pdbB[0]['xyz']
        mask[0,:NAstart,:14] = pdbA[0]['mask']
        mask[0,NAstart:,:23] = pdbB[0]['mask']
    elif (len(pdb_ids)==3):
        xyz[:,:NAstart,:14] = pdbA[0]['xyz'][None,...]
        xyz[0,NAstart:,:23] = torch.cat((pdbB[0]['xyz'], pdbB[1]['xyz']), dim=0)
        mask[:,:NAstart,:14] = pdbA[0]['mask'][None,...]
        mask[0,NAstart:,:23] = torch.cat((pdbB[0]['mask'], pdbB[1]['mask']), dim=0)
        if (NMDLS==2): # B & C are identical
            xyz[1,NAstart:,:23] = torch.cat((pdbB[1]['xyz'], pdbB[0]['xyz']), dim=0)
            mask[1,NAstart:,:23] = torch.cat((pdbB[1]['mask'], pdbB[0]['mask']), dim=0)
    else:
        xyz[0,:NAstart,:14] = torch.cat( (pdbA[0]['xyz'], pdbA[1]['xyz']), dim=0)
        xyz[1,:NAstart,:14] = torch.cat( (pdbA[1]['xyz'], pdbA[0]['xyz']), dim=0)
        xyz[:2,NAstart:,:23] = torch.cat((pdbB[0]['xyz'], pdbB[1]['xyz']), dim=0)[None,...]
        mask[0,:NAstart,:14] = torch.cat( (pdbA[0]['mask'], pdbA[1]['mask']), dim=0)
        mask[1,:NAstart,:14] = torch.cat( (pdbA[1]['mask'], pdbA[0]['mask']), dim=0)
        mask[:2,NAstart:,:23] = torch.cat( (pdbB[0]['mask'], pdbB[1]['mask']), dim=0)[None,...]
        if (NMDLS==4): # B & C are identical
            xyz[2,:NAstart,:14] = torch.cat( (pdbA[0]['xyz'], pdbA[1]['xyz']), dim=0)
            xyz[3,:NAstart,:14] = torch.cat( (pdbA[1]['xyz'], pdbA[0]['xyz']), dim=0)
            xyz[2:,NAstart:,:23] = torch.cat((pdbB[1]['xyz'], pdbB[0]['xyz']), dim=0)[None,...]
            mask[2,:NAstart,:14] = torch.cat( (pdbA[0]['mask'], pdbA[1]['mask']), dim=0)
            mask[3,:NAstart,:14] = torch.cat( (pdbA[1]['mask'], pdbA[0]['mask']), dim=0)
            mask[2:,NAstart:,:23] = torch.cat( (pdbB[1]['mask'], pdbB[0]['mask']), dim=0)[None,...]

    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(sum(Ls))
    for i in range(1,len(Ls)):
        idx[sum(Ls[:i])] += 100

    chain_idx = torch.zeros((sum(Ls), sum(Ls))).long()
    chain_idx[:Ls[0], :Ls[0]] = 1
    if (len(pdb_ids)==4):
        chain_idx[Ls[0]:NAstart, Ls[0]:NAstart] = 1
        chain_idx[NAstart:, NAstart:] = 1
    else:
        chain_idx[Ls[0]:, Ls[0]:] = 1

    # Do cropping
    if sum(Ls) > params['CROP']:
        cropref = np.random.randint(xyz.shape[0])
        sel = get_na_crop(seq[0], xyz[cropref], mask[cropref], torch.arange(sum(Ls)), Ls, params, negative)

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        mask_t = mask_t[:,sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, negative

def loader_rna(pdb_set, Ls, params, random_noise=5.0):
    # read PDBs
    pdb_ids = pdb_set.split(':')
    pdbA = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.pt')
    pdbB = None
    if (len(pdb_ids)==2):
        pdbB = torch.load(params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.pt')

    # msa for NA is sequence only
    msaA,insA = parse_fasta_if_exists(pdbA['seq'], params['NA_DIR']+'/torch/'+pdb_ids[0][1:3]+'/'+pdb_ids[0]+'.afa', rmsa_alphabet=True)
    a3m = {'msa':torch.from_numpy(msaA), 'ins':torch.from_numpy(insA)}
    if (len(pdb_ids)==2):
        msaB,insB = parse_fasta_if_exists(pdbB['seq'], params['NA_DIR']+'/torch/'+pdb_ids[1][1:3]+'/'+pdb_ids[1]+'.afa', rmsa_alphabet=True)
        a3mB = {'msa':torch.from_numpy(msaB), 'ins':torch.from_numpy(insB)}
        a3m = merge_a3m_hetero(a3m, a3mB, Ls)

    # get template features -- None
    L = sum(Ls)
    xyz_t = INIT_NA_CRDS.reshape(1,1,NTOTAL,3).repeat(1,L,1,1) + torch.rand(1,L,1,3)*random_noise
    f1d_t = torch.nn.functional.one_hot(torch.full((1, L), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
    mask_t = torch.full((1,L,NTOTAL), False)
    conf = torch.zeros((1,L,1)).float() # zero confidence
    f1d_t = torch.cat((f1d_t, conf), -1)

    NMDLS = pdbA['xyz'].shape[0]

    # get MSA features
    msa = a3m['msa'].long()
    ins = a3m['ins'].long()
    seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(msa, ins, params, L_s=Ls)

    xyz = torch.full((NMDLS, L, NTOTAL, 3), np.nan)
    mask = torch.full((NMDLS, L, NTOTAL), False)
    if (len(pdb_ids)==2):
        xyz[:,:,:23] = torch.cat((pdbA['xyz'], pdbB['xyz']), dim=1)
        mask[:,:,:23] = torch.cat((pdbA['mask'], pdbB['mask']), dim=1)
    else:
        xyz[:,:,:23] = pdbA['xyz']
        mask[:,:,:23] = pdbA['mask']
    xyz = torch.nan_to_num(xyz)

    idx = torch.arange(L)
    if (len(pdb_ids)==2):
        idx[Ls[0]:] += 100

    chain_idx = torch.ones(L,L).long()
    init = INIT_NA_CRDS.reshape(1, NTOTAL, 3).repeat(L, 1, 1)

    # Do cropping
    if sum(Ls) > params['CROP']:
        cropref = np.random.randint(xyz.shape[0])
        sel = get_na_crop(seq[0], xyz[cropref], mask[cropref], torch.arange(L), Ls, params, incl_protein=False)

        seq = seq[:,sel]
        msa_seed_orig = msa_seed_orig[:,:,sel]
        msa_seed = msa_seed[:,:,sel]
        msa_extra = msa_extra[:,:,sel]
        mask_msa = mask_msa[:,:,sel]
        xyz = xyz[:,sel]
        mask = mask[:,sel]
        xyz_t = xyz_t[:,sel]
        f1d_t = f1d_t[:,sel]
        mask_t = mask_t[:,sel]
        #
        idx = idx[sel]
        chain_idx = chain_idx[sel][:,sel]
        init = init[sel]

    xyz_prev = xyz_t[0].clone()
    mask_prev = mask_t[0].clone()

    #print ('done loader_rna',pdb_set)
    return seq.long(), msa_seed_orig.long(), msa_seed.float(), msa_extra.float(), mask_msa,\
           xyz.float(), mask, idx.long(), \
           xyz_t.float(), f1d_t.float(), mask_t, \
           xyz_prev.float(), mask_prev, \
           chain_idx, False, False

class Dataset(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, homo, unclamp_cut=0.9, pick_top=True, p_homo_cut=-1.0):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.homo = homo
        self.pick_top = pick_top
        self.unclamp_cut = unclamp_cut
        self.p_homo_cut = p_homo_cut

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        p_unclamp = np.random.rand()
        if p_unclamp > self.unclamp_cut:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo,
                              unclamp=True, 
                              pick_top=self.pick_top, 
                              p_homo_cut=self.p_homo_cut)
        else:
            out = self.loader(self.item_dict[ID][sel_idx][0], self.params, self.homo, 
                              pick_top=self.pick_top,
                              p_homo_cut=self.p_homo_cut)
        return out

class DatasetComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(self.item_dict[ID][sel_idx][0],
                          self.item_dict[ID][sel_idx][1],
                          self.item_dict[ID][sel_idx][2],
                          self.item_dict[ID][sel_idx][3],
                          self.params,
                          pick_top = self.pick_top,
                          negative = self.negative)
        return out

class DatasetNAComplex(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params, pick_top=True, negative=False, native_NA_frac=0.0):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params
        self.pick_top = pick_top
        self.negative = negative
        self.native_NA_frac = native_NA_frac

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][1],
            self.item_dict[ID][sel_idx][2],
            self.params,
            pick_top = self.pick_top,
            negative = self.negative,
            native_NA_frac = self.native_NA_frac
        )
        return out

class DatasetRNA(data.Dataset):
    def __init__(self, IDs, loader, item_dict, params):
        self.IDs = IDs
        self.item_dict = item_dict
        self.loader = loader
        self.params = params

    def __len__(self):
        return len(self.IDs)

    def __getitem__(self, index):
        ID = self.IDs[index]
        sel_idx = np.random.randint(0, len(self.item_dict[ID]))
        out = self.loader(
            self.item_dict[ID][sel_idx][0],
            self.item_dict[ID][sel_idx][1],
            self.params
        )
        return out


class DistilledDataset(data.Dataset):
    def __init__(
        self,
        pdb_IDs, pdb_loader, pdb_dict,
        compl_IDs, compl_loader, compl_dict,
        neg_IDs, neg_loader, neg_dict,
        na_compl_IDs, na_compl_loader, na_compl_dict,
        na_neg_IDs, na_neg_loader, na_neg_dict,
        fb_IDs, fb_loader, fb_dict,
        rna_IDs, rna_loader, rna_dict,
        homo, 
        params,
        native_NA_frac=0.25,
        unclamp_cut=0.9
    ):
        #
        self.pdb_IDs = pdb_IDs
        self.pdb_dict = pdb_dict
        self.pdb_loader = pdb_loader
        self.compl_IDs = compl_IDs
        self.compl_loader = compl_loader
        self.compl_dict = compl_dict
        self.neg_IDs = neg_IDs
        self.neg_loader = neg_loader
        self.neg_dict = neg_dict
        self.na_compl_IDs = na_compl_IDs
        self.na_compl_loader = na_compl_loader
        self.na_compl_dict = na_compl_dict
        self.na_neg_IDs = na_neg_IDs
        self.na_neg_loader = na_neg_loader
        self.na_neg_dict = na_neg_dict
        self.fb_IDs = fb_IDs
        self.fb_dict = fb_dict
        self.fb_loader = fb_loader
        self.rna_IDs = rna_IDs
        self.rna_dict = rna_dict
        self.rna_loader = rna_loader
        self.homo = homo
        self.params = params
        self.unclamp_cut = unclamp_cut
        self.native_NA_frac = native_NA_frac

        self.compl_inds = np.arange(len(self.compl_IDs))
        self.neg_inds = np.arange(len(self.neg_IDs))
        self.na_compl_inds = np.arange(len(self.na_compl_IDs))
        self.na_neg_inds = np.arange(len(self.na_neg_IDs))
        self.fb_inds = np.arange(len(self.fb_IDs))
        self.pdb_inds = np.arange(len(self.pdb_IDs))
        self.rna_inds = np.arange(len(self.rna_IDs))

    def __len__(self):
        return (
            len(self.fb_inds)
            + len(self.pdb_inds)
            + len(self.compl_inds)
            + len(self.neg_inds)
            + len(self.na_compl_inds)
            + len(self.na_neg_inds)
            + len(self.rna_inds)
        )

    # order:
    #    0          - nfb-1        = FB
    #    nfb        - nfb+npdb-1   = PDB
    #    "+npdb     - "+ncmpl-1    = COMPLEX
    #    "+ncmpl    - "+nneg-1     = COMPLEX NEGATIVES
    #    "+nneg     - "+nna_cmpl-1 = NA COMPLEX
    #    "+nna_cmpl - "+nrna-1     = NA COMPLEX NEGATIVES
    #    "+nrna-1   -              = RNA
    def __getitem__(self, index):
        p_unclamp = np.random.rand()

        if index < len(self.fb_inds):
            ID = self.fb_IDs[index]
            sel_idx = np.random.randint(0, len(self.fb_dict[ID]))
            out = self.fb_loader(self.fb_dict[ID][sel_idx][0], self.params, unclamp=(p_unclamp > self.unclamp_cut))

        offset = len(self.fb_inds)
        if index >= offset and index < offset + len(self.pdb_inds):
            ID = self.pdb_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.pdb_dict[ID]))
            out = self.pdb_loader(self.pdb_dict[ID][sel_idx][0], self.params, self.homo, unclamp=(p_unclamp > self.unclamp_cut))

        offset += len(self.pdb_inds)
        if index >= offset and index < offset + len(self.compl_inds):
            ID = self.compl_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.compl_dict[ID]))
            out = self.compl_loader(
                self.compl_dict[ID][sel_idx][0], 
                self.compl_dict[ID][sel_idx][1],
                self.compl_dict[ID][sel_idx][2], 
                self.compl_dict[ID][sel_idx][3], 
                self.params,
                negative=False
            )

        offset += len(self.compl_inds)
        if index >= offset and index < offset + len(self.neg_inds):
            ID = self.neg_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.neg_dict[ID]))
            out = self.neg_loader(
                self.neg_dict[ID][sel_idx][0],
                self.neg_dict[ID][sel_idx][1],
                self.neg_dict[ID][sel_idx][2],
                self.neg_dict[ID][sel_idx][3],
                self.params,
                negative=True
            )

        offset += len(self.neg_inds)
        if index >= offset and index < offset + len(self.na_compl_inds):
            ID = self.na_compl_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.na_compl_dict[ID]))
            out = self.na_compl_loader(
                self.na_compl_dict[ID][sel_idx][0],
                self.na_compl_dict[ID][sel_idx][1],
                self.na_compl_dict[ID][sel_idx][2],
                self.params,
                negative=False,
                native_NA_frac=self.native_NA_frac
            )

        offset += len(self.na_compl_inds)
        if index >= offset and index < offset + len(self.na_neg_inds):
            ID = self.na_neg_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.na_neg_dict[ID]))
            out = self.na_neg_loader(
                self.na_neg_dict[ID][sel_idx][0],
                self.na_neg_dict[ID][sel_idx][1],
                self.na_neg_dict[ID][sel_idx][2],
                self.params,
                negative=True,
                native_NA_frac=self.native_NA_frac
            )

        offset += len(self.na_neg_inds)
        if index >= offset:
            ID = self.rna_IDs[index-offset]
            sel_idx = np.random.randint(0, len(self.rna_dict[ID]))
            out = self.rna_loader(
                self.rna_dict[ID][sel_idx][0],
                self.rna_dict[ID][sel_idx][1],
                self.params
            )

        return out

class DistributedWeightedSampler(data.Sampler):
    def __init__(
        self,
        dataset,
        pdb_weights,
        fb_weights,
        compl_weights,
        neg_weights,
        na_compl_weights,
        neg_na_compl_weights,
        rna_weights,
        num_example_per_epoch=25600,
        fraction_fb=0.2,
        fraction_compl=0.2,  # half neg, half pos
        fraction_na_compl=0.2, # half neg, half pos
        fraction_rna=0.2,
        num_replicas=None,
        rank=None,
        replacement=False
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        assert num_example_per_epoch % num_replicas == 0
        assert (fraction_fb+fraction_compl+fraction_na_compl+fraction_rna <= 1.0)

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.num_fb_per_epoch = int(round(num_example_per_epoch*fraction_fb))
        self.num_compl_per_epoch = int(round(0.5*num_example_per_epoch*fraction_compl))
        self.num_neg_per_epoch = self.num_compl_per_epoch
        self.num_na_compl_per_epoch = int(round(0.5*num_example_per_epoch*fraction_na_compl))
        self.num_neg_na_compl_per_epoch = self.num_na_compl_per_epoch
        self.num_rna_per_epoch = int(round(num_example_per_epoch*fraction_rna))

        self.num_pdb_per_epoch = num_example_per_epoch - (
            self.num_fb_per_epoch 
            + self.num_compl_per_epoch
            + self.num_neg_per_epoch
            + self.num_na_compl_per_epoch
            + self.num_neg_na_compl_per_epoch
            + self.num_rna_per_epoch
        )

        if (rank==0):
            print (
                "Per epoch:",
                self.num_pdb_per_epoch,"pdb,",
                self.num_fb_per_epoch,"fb,",
                self.num_compl_per_epoch,"compl,",
                self.num_neg_per_epoch,"neg,",
                self.num_na_compl_per_epoch,"NA compl,",
                self.num_neg_na_compl_per_epoch,"NA neg,",
                self.num_rna_per_epoch,"RNA."
            )


        self.total_size = num_example_per_epoch
        self.num_samples = self.total_size // self.num_replicas
        self.rank = rank
        self.epoch = 0
        self.replacement = replacement

        self.pdb_weights = pdb_weights
        self.fb_weights = fb_weights

        self.compl_weights = compl_weights
        self.neg_weights = neg_weights

        self.na_compl_weights = na_compl_weights
        self.neg_na_compl_weights = neg_na_compl_weights

        self.rna_weights = rna_weights

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        # get indices (fb + pdb models)
        indices = torch.arange(len(self.dataset))

        # weighted subsampling
        # order:
        #    0          - nfb-1        = FB
        #    nfb        - nfb+npdb-1   = PDB
        #    "+npdb     - "+ncmpl-1    = COMPLEX
        #    "+ncmpl    - "+nneg-1     = COMPLEX NEGATIVES
        #    "+nneg     - "+nna_cmpl-1 = NA COMPLEX
        #    "+nna_cmpl - "+nrna-1     = NA COMPLEX NEGATIVES
        #    "+nrna-1   -              = RNA
        sel_indices = torch.tensor((),dtype=int)
        if (self.num_fb_per_epoch>0):
            fb_sampled = torch.multinomial(self.fb_weights, self.num_fb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[fb_sampled]))

        if (self.num_pdb_per_epoch>0):
            offset = len(self.dataset.fb_IDs)
            pdb_sampled = torch.multinomial(self.pdb_weights, self.num_pdb_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[pdb_sampled + offset]))

        if (self.num_compl_per_epoch>0):
            offset = len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs)
            compl_sampled = torch.multinomial(self.compl_weights, self.num_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[compl_sampled + offset]))
        
        if (self.num_neg_per_epoch>0):
            offset = len(self.dataset.fb_IDs) + len(self.dataset.pdb_IDs) + len(self.dataset.compl_IDs)
            neg_sampled = torch.multinomial(self.neg_weights, self.num_neg_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[neg_sampled + offset]))

        if (self.num_na_compl_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
            )
            na_compl_sampled = torch.multinomial(self.na_compl_weights, self.num_na_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[na_compl_sampled + offset]))

        if (self.num_neg_na_compl_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
                + len(self.dataset.na_compl_IDs)
            )
            neg_na_sampled = torch.multinomial(self.neg_na_compl_weights, self.num_neg_na_compl_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[neg_na_sampled + offset]))

        if (self.num_rna_per_epoch>0):
            offset = (
                len(self.dataset.fb_IDs) 
                + len(self.dataset.pdb_IDs) 
                + len(self.dataset.compl_IDs)
                + len(self.dataset.neg_IDs)
                + len(self.dataset.na_compl_IDs)
                + len(self.dataset.na_neg_IDs)
            )
            rna_sampled = torch.multinomial(self.rna_weights, self.num_rna_per_epoch, self.replacement, generator=g)
            sel_indices = torch.cat((sel_indices, indices[rna_sampled + offset]))

        # shuffle indices
        indices = sel_indices[torch.randperm(len(sel_indices), generator=g)]

        # per each gpu
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices.tolist())

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


