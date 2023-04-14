import sys, os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
from parsers import parse_a3m, parse_fasta, parse_mixed_fasta, read_template_pdb, parse_pdb_w_seq, read_templates
from RoseTTAFoldModel  import RoseTTAFoldModule
import util
from collections import namedtuple
from ffindex import *
from data_loader import MSAFeaturize, MSABlockDeletion, merge_a3m_homo, merge_a3m_hetero
from kinematics import xyz_to_c6d, c6d_to_bins, xyz_to_t2d
from util_module import XYZConverter
from chemical import NTOTAL, NTOTALDOFS, NAATOKENS, INIT_CRDS, INIT_NA_CRDS

# suppress dgl warning w/ newest pytorch
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

def get_args():
    import argparse
    parser = argparse.ArgumentParser(description="RoseTTAFold2NA")
    parser.add_argument("-inputs", help="R|Input data in format A:B:C:D, with\n"
         "   A = P or D or S or R or PR - fasta is protein, dsDNA, ssDNA, RNA, or coupled protein/RNA\n"
         "   B = multiple sequence alignment file (a3m for protein, afa for RNA, fasta for DNA)\n"
         "   C = hhpred hhr file\n"
         "   D = hhpred atab file\n"
         "Spaces seperate multiple inputs.  The last two arguments may be omitted\n",
         default=None, nargs='+')
    parser.add_argument("-db", help="HHpred database location", default=None)
    parser.add_argument("-prefix", help="output prefix", type=str, default="S")
    parser.add_argument("-model", default=None, help="The model weights")
    args = parser.parse_args()
    return args

MAX_CYCLE = 10
NMODELS = 1
NBIN = [37, 37, 37, 19]

MAXLAT = 256
MAXSEQ = 2048

MODEL_PARAM ={
        "n_extra_block"   : 4,
        "n_main_block"    : 32,
        "n_ref_block"     : 4,
        "d_msa"           : 256 ,
        "d_pair"          : 128,
        "d_templ"         : 64,
        "n_head_msa"      : 8,
        "n_head_pair"     : 4,
        "n_head_templ"    : 4,
        "d_hidden"        : 32,
        "d_hidden_templ"  : 64,
        "p_drop"       : 0.0,
        "lj_lin"       : 0.75
}

SE3_param = {
        "num_layers"    : 1,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
}

SE3_ref_param = {
        "num_layers"    : 2,
        "num_channels"  : 32,
        "num_degrees"   : 2,
        "l0_in_features": 64,
        "l0_out_features": 64,
        "l1_in_features": 3,
        "l1_out_features": 2,
        "num_edge_features": 64,
        "div": 4,
        "n_heads": 4
}

MODEL_PARAM['SE3_param_full'] = SE3_param
MODEL_PARAM['SE3_param_topk'] = SE3_ref_param

def lddt_unbin(pred_lddt):
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    
    pred_lddt = nn.Softmax(dim=1)(pred_lddt)
    return torch.sum(lddt_bins[None,:,None]*pred_lddt, dim=1)

def pae_unbin(pred_pae):
    # calculate pae loss
    nbin = pred_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin, dtype=pred_pae.dtype, device=pred_pae.device)

    pred_pae = nn.Softmax(dim=1)(pred_pae)
    return torch.sum(pae_bins[None,:,None,None]*pred_pae, dim=1)

class Predictor():
    def __init__(self, model_weights, device="cuda:0"):
        # define model name
        self.model_weights = model_weights
        self.device = device
        self.active_fn = nn.Softmax(dim=1)

        # define model & load model
        self.model = RoseTTAFoldModule(
            **MODEL_PARAM,
            aamask=util.allatom_mask.to(self.device),
            ljlk_parameters=util.ljlk_parameters.to(self.device),
            lj_correction_parameters=util.lj_correction_parameters.to(self.device),
            num_bonds=util.num_bonds.to(self.device),
            hbtypes=util.hbtypes.to(self.device),
            hbbaseatoms=util.hbbaseatoms.to(self.device),
            hbpolys=util.hbpolys.to(self.device)
        ).to(self.device)

        could_load = self.load_model(self.model_weights)
        if not could_load:
            print ("ERROR: failed to load model")
            sys.exit()

        self.xyz_converter = XYZConverter()

    def load_model(self, model_weights):
        if not os.path.exists(model_weights):
            return False
        checkpoint = torch.load(model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return True

    def predict(self, inputs, out_prefix, ffdb, n_templ=4):
        # pass 1, combined MSA
        Ls, msas, inss, types = [], [], [], []
        has_paired = False
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            fseq_i[0] = fseq_i[0].upper()
            assert (len(fseq_i) >= 2)
            assert (fseq_i[0] in ["P","R","D","S","PR"])
            type_i,a3m_i = fseq_i[:2]

            if (fseq_i[0]=="PR"):
                msa_i, ins_i, Ls_i = parse_mixed_fasta(a3m_i)
                Ls.extend(Ls_i)
                has_paired = True
            else:
                if (fseq_i[0]=="P"):
                    msa_i, ins_i = parse_a3m(a3m_i)
                else:
                    is_rna = fseq_i[0]=='R'
                    is_dna = fseq_i[0]=='D' or fseq_i[0]=='S'
                    msa_i, ins_i = parse_fasta(a3m_i, rna_alphabet=is_rna, dna_alphabet=is_dna)

                _, L = msa_i.shape
                Ls.append(L)

            msa_i = torch.tensor(msa_i).long()
            ins_i = torch.tensor(ins_i).long()

            if (msa_i.shape[0] > MAXSEQ):
                idxs_tokeep = np.random.permutation(msa_i.shape[0])[:MAXSEQ]
                idxs_tokeep[0] = 0  # keep best
                msa_i = msa_i[idxs_tokeep]
                ins_i = ins_i[idxs_tokeep]

            msas.append(msa_i)
            inss.append(ins_i)
            types.append(fseq_i[0])

            # add strand compliment
            if (fseq_i[0]=='D'):
                msas.append( util.dna_reverse_complement(msa_i) )
                inss.append( ins_i.clone() )
                Ls.append(L)
                types.append(fseq_i[0])

        msa_orig = {'msa':msas[0],'ins':inss[0]}
        if (has_paired):
            if (len(Ls)!=2 or len(msas)!=1):
                print ("ERROR: Paired Protein/NA fastas can not be combined with other inputs!")
                assert (False)
        else:
            for i in range(1,len(Ls)):
                msa_orig = merge_a3m_hetero(msa_orig, {'msa':msas[i],'ins':inss[i]}, [sum(Ls[:i]),Ls[i]])

        msa_orig, ins_orig = msa_orig['msa'], msa_orig['ins']

        # pass 2, templates
        L = sum(Ls)
        xyz_t = INIT_CRDS.reshape(1,1,NTOTAL,3).repeat(n_templ,L,1,1) + torch.rand(n_templ,L,1,3)*5.0 - 2.5
        is_NA = util.is_nucleic(msa_orig[0])
        xyz_t[:,is_NA] = INIT_NA_CRDS.reshape(1,1,NTOTAL,3)

        mask_t = torch.full((n_templ, L, NTOTAL), False) 
        t1d = torch.nn.functional.one_hot(torch.full((n_templ, L), 20).long(), num_classes=NAATOKENS-1).float() # all gaps
        t1d = torch.cat((t1d, torch.zeros((n_templ,L,1)).float()), -1)

        maxtmpl=1
        for i,seq_i in enumerate(inputs):
            fseq_i =  seq_i.split(':')
            fseq_i[0] = fseq_i[0].upper()
            if (fseq_i[0]=="P" and len(fseq_i) == 4):
                hhr_i,atab_i = fseq_i[2:4]
                startres,stopres = sum(Ls[:i]), sum(Ls[:(i+1)])
                xyz_t_i, t1d_i, mask_t_i = read_templates(Ls[i], ffdb, hhr_i, atab_i, n_templ=n_templ)
                ntmpl_i = xyz_t_i.shape[0]
                maxtmpl = max(maxtmpl, ntmpl_i)
                xyz_t[:ntmpl_i,startres:stopres,:,:] = xyz_t_i
                t1d[:ntmpl_i,startres:stopres,:] = t1d_i
                mask_t[:ntmpl_i,startres:stopres,:] = mask_t_i

        same_chain = torch.zeros((1,L,L), dtype=torch.bool, device=xyz_t.device)
        stopres = 0
        for i in range(1,len(Ls)):
            startres,stopres = sum(Ls[:(i-1)]), sum(Ls[:i])
            same_chain[:,startres:stopres,startres:stopres] = True
        same_chain[:,stopres:,stopres:] = True

        # template features
        xyz_t = xyz_t[:maxtmpl].float().unsqueeze(0)
        mask_t = mask_t[:maxtmpl].unsqueeze(0)
        t1d = t1d[:maxtmpl].float().unsqueeze(0)

        mask_t_2d = mask_t[:,:,:,:3].all(dim=-1) # (B, T, L)
        mask_t_2d = mask_t_2d[:,:,None]*mask_t_2d[:,:,:,None] # (B, T, L, L)
        mask_t_2d = mask_t_2d.float()*same_chain.float()[:,None] # (ignore inter-chain region)
        t2d = xyz_to_t2d(xyz_t, mask_t_2d)

        seq_tmp = t1d[...,:-1].argmax(dim=-1).reshape(-1,L)
        alpha, _, alpha_mask, _ = self.xyz_converter.get_torsions(xyz_t.reshape(-1,L,NTOTAL,3), seq_tmp, mask_in=mask_t.reshape(-1,L,NTOTAL))
        alpha_mask = torch.logical_and(alpha_mask, ~torch.isnan(alpha[...,0]))

        alpha[torch.isnan(alpha)] = 0.0
        alpha = alpha.reshape(1,-1,L,NTOTALDOFS,2)
        alpha_mask = alpha_mask.reshape(1,-1,L,NTOTALDOFS,1)
        alpha_t = torch.cat((alpha, alpha_mask), dim=-1).reshape(1, -1, L, 3*NTOTALDOFS)

        self.model.eval()
        for i_trial in range(NMODELS):
            if os.path.exists("%s_%02d.pdb"%(out_prefix, i_trial)):
                continue
            self._run_model(Ls, msa_orig, ins_orig, t1d, t2d, xyz_t, xyz_t[:,0], alpha_t, same_chain, mask_t_2d, "%s_%02d"%(out_prefix, i_trial))
            torch.cuda.empty_cache()

    def _run_model(self, L_s, msa_orig, ins_orig, t1d, t2d, xyz_t, xyz, alpha_t, same_chain, mask_t_2d, out_prefix):
        self.xyz_converter = self.xyz_converter.to(self.device)
        with torch.no_grad():
            seq, msa_seed_orig, msa_seed, msa_extra, mask_msa = MSAFeaturize(
                msa_orig, ins_orig, p_mask=0.0, params={'MAXLAT': MAXLAT, 'MAXSEQ': MAXSEQ, 'MAXCYCLE': MAX_CYCLE})

            _, N, L = msa_seed.shape[:3]
            B = 1   
            #
            idx_pdb = torch.arange(L).long().view(1, L)
            for i in range(len(L_s)-1):
                idx_pdb[ :, sum(L_s[:(i+1)]): ] += 100

            #
            seq = seq.unsqueeze(0)
            msa_seed = msa_seed.unsqueeze(0)
            msa_extra = msa_extra.unsqueeze(0)

            t1d = t1d.to(self.device)
            t2d = t2d.to(self.device)
            idx_pdb = idx_pdb.to(self.device)
            xyz_t = xyz_t.to(self.device)
            alpha_t = alpha_t.to(self.device)
            xyz = xyz.to(self.device)
            same_chain = same_chain.to(self.device)
            mask_t_2d = mask_t_2d.to(self.device)

            msa_prev = None
            pair_prev = None
            alpha_prev = torch.zeros((1,L,NTOTALDOFS,2), device=self.device)
            xyz_prev=xyz
            state_prev = None

            best_lddt = torch.tensor([-1.0], device=self.device)
            best_xyz = None
            best_logit = None
            best_aa = None
            print ("           plddt    best")
            for i_cycle in range(MAX_CYCLE):
                msa_seed_i = msa_seed[:,i_cycle].to(self.device)
                msa_extra_i = msa_extra[:,i_cycle].to(self.device)
                seq_i = seq[:,i_cycle].to(self.device)
                with torch.cuda.amp.autocast(True):
                    logit_s, logit_aa_s, logit_pae, p_bind, init_crds, alpha_prev, _, pred_lddt_binned, msa_prev, pair_prev, state_prev = self.model(
                        msa_latent=msa_seed_i, 
                        msa_full=msa_extra_i,
                        seq=seq_i, 
                        seq_unmasked=seq_i, 
                        xyz=xyz_prev, 
                        sctors=alpha_prev,
                        idx=idx_pdb,
                        t1d=t1d, 
                        t2d=t2d,
                        xyz_t=xyz_t[:,:,:,1],
                        mask_t=mask_t_2d,
                        alpha_t=alpha_t,
                        msa_prev=msa_prev,
                        pair_prev=pair_prev,
                        state_prev=state_prev,
                        same_chain=same_chain
                    )

                    logit_aa_s = logit_aa_s.reshape(B,-1,N,L)[:,:,0].permute(0,2,1)

                xyz_prev = init_crds[-1]
                alpha_prev = alpha_prev[-1]
                pred_lddt = lddt_unbin(pred_lddt_binned)
                pae = pae_unbin(logit_pae)

                print ("RECYCLE %2d %7.3f %7.3f"%( 
                    i_cycle, 
                    pred_lddt.mean().cpu().numpy(), 
                    best_lddt.mean().cpu().numpy()
                ) )

                _, all_crds = self.xyz_converter.compute_all_atom(seq[:,i_cycle], init_crds[-1], alpha_prev)

                if pred_lddt.mean() < best_lddt.mean():
                    continue

                best_xyz = all_crds.clone()
                best_logit = logit_s
                best_aa = logit_aa_s
                best_lddt = pred_lddt.clone()
                best_pae = pae.clone()

            prob_s = list()
            for logit in logit_s:
                prob = self.active_fn(logit.float()) # distogram
                prob = prob.reshape(-1, L, L) #.permute(1,2,0).cpu().numpy()
                prob_s.append(prob)
        
        end = time.time()

        for prob in prob_s:
            prob += 1e-8
            prob = prob / torch.sum(prob, dim=0)[None]
        util.writepdb(out_prefix+".pdb", best_xyz[0], seq[0, -1], L_s, bfacts=100*best_lddt[0].float())
        prob_s = [prob.permute(1,2,0).detach().cpu().numpy().astype(np.float16) for prob in prob_s]
        np.savez_compressed("%s.npz"%(out_prefix), 
            dist=prob_s[0].astype(np.float16), \
            lddt=best_lddt[0].detach().cpu().numpy().astype(np.float16), \
            pae=best_pae[0].detach().cpu().numpy().astype(np.float16)
        )

if __name__ == "__main__":
    args = get_args()

    # Read template database
    FFDB = args.db
    FFindexDB = namedtuple("FFindexDB", "index, data")
    ffdb = FFindexDB(read_index(FFDB+'_pdb.ffindex'),
                     read_data(FFDB+'_pdb.ffdata'))

    if (torch.cuda.is_available()):
        print ("Running on GPU")
        pred = Predictor(args.model, torch.device("cuda:0"))
    else:
        print ("Running on CPU")
        pred = Predictor(args.model, torch.device("cpu"))

    pred.predict(inputs=args.inputs, out_prefix=args.prefix, ffdb=ffdb)
