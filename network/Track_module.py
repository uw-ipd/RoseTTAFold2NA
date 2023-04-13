import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum
import torch.utils.checkpoint as checkpoint
from util_module import *
from Attention_module import *
from SE3_network import SE3TransformerWrapper
from loss import (
    calc_BB_bond_geom_grads, calc_lj_grads, calc_hb_grads, calc_lj
)
from chemical import NTOTALDOFS

# Module contains classes and functions to generate initial embeddings
class PositionalEncoding2D(nn.Module):
    # Add relative positional encoding to pair features
    def __init__(self, d_model, minpos=-32, maxpos=32):
        super(PositionalEncoding2D, self).__init__()
        self.minpos = minpos
        self.maxpos = maxpos
        self.nbin = abs(minpos)+maxpos+1
        self.emb = nn.Embedding(self.nbin, d_model)
        self.emb_chain = nn.Embedding(2, d_model)
    
    def forward(self, idx, same_chain):
        B, L = idx.shape[:2]
        bins = torch.arange(self.minpos, self.maxpos, device=idx.device)
        seqsep = idx[:,None,:] - idx[:,:,None] # (B, L, L)
        #
        ib = torch.bucketize(seqsep, bins).long() # (B, L, L)
        emb = self.emb(ib) #(B, L, L, d_model)
        emb_c = self.emb_chain(same_chain.long())
        return emb + emb_c


# Components for three-track blocks
# 1. MSA -> MSA update (biased attention. bias from pair & structure)
# 2. Pair -> Pair update (biased attention. bias from structure)
# 3. MSA -> Pair update (extract coevolution signal)
# 4. Str -> Str update (node from MSA, edge from Pair)

# Update MSA with biased self-attention. bias from Pair & Str
class MSAPairStr2MSA(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, n_head=8, d_state=16, d_rbf=64, 
                 d_hidden=32, p_drop=0.15, use_global_attn=False):
        super(MSAPairStr2MSA, self).__init__()
        self.norm_pair = nn.LayerNorm(d_pair)

        self.emb_rbf = nn.Linear(d_rbf, d_pair) # new in v2

        self.norm_state = nn.LayerNorm(d_state)
        self.proj_state = nn.Linear(d_state, d_msa)
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.row_attn = MSARowAttentionWithBias(d_msa=d_msa, d_pair=d_pair,
                                                n_head=n_head, d_hidden=d_hidden) 
        if use_global_attn:
            self.col_attn = MSAColGlobalAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        else:
            self.col_attn = MSAColAttention(d_msa=d_msa, n_head=n_head, d_hidden=d_hidden) 
        self.ff = FeedForwardLayer(d_msa, 4, p_drop=p_drop)
        
        # Do proper initialization
        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distrib
        self.emb_rbf = init_lecun_normal(self.emb_rbf)
        self.proj_state = init_lecun_normal(self.proj_state)

        # initialize bias to zeros
        nn.init.zeros_(self.emb_rbf.bias)
        nn.init.zeros_(self.proj_state.bias)

    def forward(self, msa, pair, rbf_feat, state):
        '''
        Inputs:
            - msa: MSA feature (B, N, L, d_msa)
            - pair: Pair feature (B, L, L, d_pair)
            - rbf_feat: Ca-Ca distance feature calculated from xyz coordinates (B, L, L, d_rbf)
            - xyz: xyz coordinates (B, L, n_atom, 3)
            - state: updated node features after SE(3)-Transformer layer (B, L, d_state)
        Output:
            - msa: Updated MSA feature (B, N, L, d_msa)
        '''
        B, N, L = msa.shape[:3]

        # prepare input bias feature by combining pair & coordinate info
        pair = self.norm_pair(pair)
        pair = pair + self.emb_rbf(rbf_feat)

        #neighbor = get_seqsep(idx)
        #pair = pair + self.emb_seqsep(neighbor)

        #
        # update query sequence feature (first sequence in the MSA) with feedbacks (state) from SE3
        state = self.norm_state(state)
        state = self.proj_state(state).reshape(B, 1, L, -1)
        msa = msa.type_as(state)
        msa = msa.index_add(1, torch.tensor([0,], device=state.device), state)
        #
        # Apply row/column attention to msa & transform 
        msa = msa + self.drop_row(self.row_attn(msa, pair))
        msa = msa + self.col_attn(msa)
        msa = msa + self.ff(msa)

        return msa

class PairStr2Pair(nn.Module):
    def __init__(self, d_pair=128, n_head=4, d_hidden=32, d_rbf=64, p_drop=0.15):
        super(PairStr2Pair, self).__init__()
        
        self.drop_row = Dropout(broadcast_dim=1, p_drop=p_drop)
        self.drop_col = Dropout(broadcast_dim=2, p_drop=p_drop)

        self.row_attn = BiasedAxialAttention(d_pair, d_rbf, n_head, d_hidden, p_drop=p_drop, is_row=True)
        self.col_attn = BiasedAxialAttention(d_pair, d_rbf, n_head, d_hidden, p_drop=p_drop, is_row=False)

        self.ff = FeedForwardLayer(d_pair, 2)

    def forward(self, pair, rbf_feat):
        B, L = pair.shape[:2]

        pair = pair + self.drop_row(self.row_attn(pair, rbf_feat))
        pair = pair + self.drop_col(self.col_attn(pair, rbf_feat))
        pair = pair + self.ff(pair)
        return pair

class MSA2Pair(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_hidden=32, p_drop=0.15):
        super(MSA2Pair, self).__init__()
        self.norm = nn.LayerNorm(d_msa)
        self.proj_left = nn.Linear(d_msa, d_hidden)
        self.proj_right = nn.Linear(d_msa, d_hidden)
        self.proj_out = nn.Linear(d_hidden*d_hidden, d_pair)
        
        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.proj_left = init_lecun_normal(self.proj_left)
        self.proj_right = init_lecun_normal(self.proj_right)
        nn.init.zeros_(self.proj_left.bias)
        nn.init.zeros_(self.proj_right.bias)

        # zero initialize output
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, msa, pair):
        B, N, L = msa.shape[:3]
        msa = self.norm(msa)
        left = self.proj_left(msa)
        right = self.proj_right(msa)
        right = right / float(N)
        out = einsum('bsli,bsmj->blmij', left, right).reshape(B, L, L, -1)
        out = self.proj_out(out)
       
        pair = pair + out
        
        return pair

class SCPred(nn.Module):
    def __init__(self, d_msa=256, d_state=32, d_hidden=128, p_drop=0.15):
        super(SCPred, self).__init__()
        self.norm_s0 = nn.LayerNorm(d_msa)
        self.norm_si = nn.LayerNorm(d_state)
        self.linear_s0 = nn.Linear(d_msa, d_hidden)
        self.linear_si = nn.Linear(d_state, d_hidden)

        # ResNet layers
        self.linear_1 = nn.Linear(d_hidden, d_hidden)
        self.linear_2 = nn.Linear(d_hidden, d_hidden)
        self.linear_3 = nn.Linear(d_hidden, d_hidden)
        self.linear_4 = nn.Linear(d_hidden, d_hidden)

        # Final outputs
        self.linear_out = nn.Linear(d_hidden, 2*NTOTALDOFS)

        self.reset_parameter()

    def reset_parameter(self):
        # normal initialization
        self.linear_s0 = init_lecun_normal(self.linear_s0)
        self.linear_si = init_lecun_normal(self.linear_si)
        self.linear_out = init_lecun_normal(self.linear_out)
        nn.init.zeros_(self.linear_s0.bias)
        nn.init.zeros_(self.linear_si.bias)
        nn.init.zeros_(self.linear_out.bias)
        
        # right before relu activation: He initializer (kaiming normal)
        nn.init.kaiming_normal_(self.linear_1.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_1.bias)
        nn.init.kaiming_normal_(self.linear_3.weight, nonlinearity='relu')
        nn.init.zeros_(self.linear_3.bias)

        # right before residual connection: zero initialize
        nn.init.zeros_(self.linear_2.weight)
        nn.init.zeros_(self.linear_2.bias)
        nn.init.zeros_(self.linear_4.weight)
        nn.init.zeros_(self.linear_4.bias)
    
    def forward(self, seq, state):
        '''
        Predict side-chain torsion angles along with backbone torsions
        Inputs:
            - seq: hidden embeddings corresponding to query sequence (B, L, d_msa)
            - state: state feature (output l0 feature) from previous SE3 layer (B, L, d_state)
        Outputs:
            - si: predicted torsion angles (phi, psi, omega, chi1~4 with cos/sin, Cb bend, Cb twist, CG) (B, L, 10, 2)
        '''
        B, L = seq.shape[:2]
        seq = self.norm_s0(seq)
        state = self.norm_si(state)
        si = self.linear_s0(seq) + self.linear_si(state)

        si = si + self.linear_2(F.relu_(self.linear_1(F.relu_(si))))
        si = si + self.linear_4(F.relu_(self.linear_3(F.relu_(si))))

        si = self.linear_out(F.relu_(si))
        return si.view(B, L, NTOTALDOFS, 2)


class Str2Str(nn.Module):
    def __init__(self, d_msa=256, d_pair=128, d_state=16, 
            SE3_param={}, 
            nextra_l0=0, nextra_l1=0,
            rbf_sigma=1.0, p_drop=0.1
    ):
        super(Str2Str, self).__init__()
        
        # initial node & pair feature process
        self.norm_msa = nn.LayerNorm(d_msa)
        self.norm_pair = nn.LayerNorm(d_pair)
        self.norm_state = nn.LayerNorm(d_state)
    
        self.embed_x = nn.Linear(d_msa+d_state, SE3_param['l0_in_features'])
        self.embed_e1 = nn.Linear(d_pair, SE3_param['num_edge_features'])
        self.embed_e2 = nn.Linear(SE3_param['num_edge_features']+64+1, SE3_param['num_edge_features'])
        
        self.norm_node = nn.LayerNorm(SE3_param['l0_in_features'])
        self.norm_edge1 = nn.LayerNorm(SE3_param['num_edge_features'])
        self.norm_edge2 = nn.LayerNorm(SE3_param['num_edge_features'])

        SE3_param_temp = SE3_param.copy()
        SE3_param_temp['l0_in_features'] += nextra_l0
        SE3_param_temp['l1_in_features'] += nextra_l1
        
        self.se3 = SE3TransformerWrapper(**SE3_param_temp)
        self.rbf_sigma = rbf_sigma
        self.sc_predictor = SCPred(
            d_msa=d_msa,
            d_state=SE3_param['l0_out_features'],
            p_drop=p_drop)

        self.reset_parameter()

    def reset_parameter(self):
        # initialize weights to normal distribution
        self.embed_x = init_lecun_normal(self.embed_x)
        self.embed_e1 = init_lecun_normal(self.embed_e1)
        self.embed_e2 = init_lecun_normal(self.embed_e2)

        # initialize bias to zeros
        nn.init.zeros_(self.embed_x.bias)
        nn.init.zeros_(self.embed_e1.bias)
        nn.init.zeros_(self.embed_e2.bias)
    
    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, msa, pair, xyz, state, idx, extra_l0=None, extra_l1=None, top_k=128, eps=1e-5):
        # process msa & pair features
        B, N, L = msa.shape[:3]
        node = self.norm_msa(msa[:,0])
        pair = self.norm_pair(pair)
        state = self.norm_state(state)

        node = torch.cat((node, state), dim=-1)
        node = self.norm_node(self.embed_x(node))

        pair = self.norm_edge1(self.embed_e1(pair))

        neighbor = get_seqsep(idx)
        rbf_feat = rbf(torch.cdist(xyz[:,:,1], xyz[:,:,1]))
        pair = torch.cat((pair, rbf_feat, neighbor), dim=-1)
        pair = self.norm_edge2(self.embed_e2(pair))

        # define graph
        if top_k > 0:
            G, edge_feats = make_topk_graph(xyz[:,:,1,:], pair, idx, top_k=top_k)
        else:
            G, edge_feats = make_full_graph(xyz[:,:,1,:], pair, idx)
        l1_feats = xyz - xyz[:,:,1,:].unsqueeze(2)
        l1_feats = l1_feats.reshape(B*L, -1, 3)
        if extra_l1 is not None:
            l1_feats = torch.cat( (l1_feats,extra_l1), dim=1 )
        if extra_l0 is not None:
            node = torch.cat( (node,extra_l0), dim=2 )

        # apply SE(3) Transformer & update coordinates
        shift = self.se3(G, node.reshape(B*L, -1, 1), l1_feats, edge_feats)

        state = shift['0'].reshape(B, L, -1) # (B, L, C)
        
        offset = shift['1'].reshape(B, L, 2, 3)
        T = offset[:,:,0,:] / 10.0
        R = offset[:,:,1,:] / 100.0

        Qnorm = torch.sqrt( 1 + torch.sum(R*R, dim=-1) )
        qA, qB, qC, qD = 1/Qnorm, R[:,:,0]/Qnorm, R[:,:,1]/Qnorm, R[:,:,2]/Qnorm

        v = xyz - xyz[:,:,1:2,:]
        Rout = torch.zeros((B,L,3,3), device=xyz.device)
        Rout[:,:,0,0] = qA*qA+qB*qB-qC*qC-qD*qD
        Rout[:,:,0,1] = 2*qB*qC - 2*qA*qD
        Rout[:,:,0,2] = 2*qB*qD + 2*qA*qC
        Rout[:,:,1,0] = 2*qB*qC + 2*qA*qD
        Rout[:,:,1,1] = qA*qA-qB*qB+qC*qC-qD*qD
        Rout[:,:,1,2] = 2*qC*qD - 2*qA*qB
        Rout[:,:,2,0] = 2*qB*qD - 2*qA*qC
        Rout[:,:,2,1] = 2*qC*qD + 2*qA*qB
        Rout[:,:,2,2] = qA*qA-qB*qB-qC*qC+qD*qD

        xyz = torch.einsum('blij,blaj->blai', Rout,v)+xyz[:,:,1:2,:]+T[:,:,None,:]

        alpha = self.sc_predictor(msa[:,0], state)

        return xyz, state, alpha


class IterBlock(nn.Module):
    def __init__(self, d_msa=256, d_pair=128,
                 n_head_msa=8, n_head_pair=4,
                 use_global_attn=False,
                 d_hidden=32, d_hidden_msa=None, p_drop=0.15,
                 SE3_param={'l0_in_features':32, 'l0_out_features':16, 'num_edge_features':32}):
        super(IterBlock, self).__init__()
        if d_hidden_msa == None:
            d_hidden_msa = d_hidden

        self.msa2msa = MSAPairStr2MSA(d_msa=d_msa, d_pair=d_pair,
                                      n_head=n_head_msa,
                                      d_state=SE3_param['l0_out_features'],
                                      use_global_attn=use_global_attn,
                                      d_hidden=d_hidden_msa, p_drop=p_drop)
        self.msa2pair = MSA2Pair(d_msa=d_msa, d_pair=d_pair,
                                 d_hidden=d_hidden//2, p_drop=p_drop)
        self.pair2pair = PairStr2Pair(d_pair=d_pair, n_head=n_head_pair, 
                                      d_hidden=d_hidden, p_drop=p_drop)
        self.str2str = Str2Str(d_msa=d_msa, d_pair=d_pair,
                               d_state=SE3_param['l0_out_features'],
                               SE3_param=SE3_param,
                               p_drop=p_drop)

    def forward(self, msa, pair, xyz, state, idx, use_checkpoint=False):
        cas = xyz[:,:,1,:].contiguous()
        rbf_feat = rbf(torch.cdist(cas, cas))
        if use_checkpoint:
            msa = checkpoint.checkpoint(create_custom_forward(self.msa2msa), msa, pair, rbf_feat, state)
            pair = checkpoint.checkpoint(create_custom_forward(self.msa2pair), msa, pair)
            pair = checkpoint.checkpoint(create_custom_forward(self.pair2pair), pair, rbf_feat)

            xyz, state, alpha = checkpoint.checkpoint(create_custom_forward(self.str2str, top_k=0), 
                msa.float(), pair.float(), xyz.detach().float(), state.float(), idx)

        else:
            msa = self.msa2msa(msa, pair, rbf_feat, state)
            pair = self.msa2pair(msa, pair)
            pair = self.pair2pair(pair, rbf_feat)

            xyz, state, alpha = self.str2str(msa.float(), pair.float(), xyz.detach().float(), state.float(), idx, top_k=0)
        
        return msa, pair, xyz, state, alpha

class IterativeSimulator(nn.Module):
    def __init__(
        self, n_extra_block=4, n_main_block=12, n_ref_block=4,
        d_msa=256, d_msa_full=64, d_pair=128, d_hidden=32,
        n_head_msa=8, n_head_pair=4,
        SE3_param_full={}, SE3_param_topk={},
        p_drop=0.15,
        aamask=None, ljlk_parameters=None, lj_correction_parameters=None, num_bonds=None, lj_lin=0.6,
        hbtypes=None, hbbaseatoms=None, hbpolys=None
    ):
        super(IterativeSimulator, self).__init__()
        self.n_extra_block = n_extra_block
        self.n_main_block = n_main_block
        self.n_ref_block = n_ref_block

        self.aamask = aamask
        self.ljlk_parameters = ljlk_parameters 
        self.lj_correction_parameters = lj_correction_parameters
        self.num_bonds = num_bonds
        self.lj_lin = lj_lin
        self.hbtypes = hbtypes 
        self.hbbaseatoms = hbbaseatoms
        self.hbpolys = hbpolys

        self.xyzconverter = XYZConverter()

        # Update with extra sequences
        if n_extra_block > 0:
            self.extra_block = nn.ModuleList([IterBlock(d_msa=d_msa_full, d_pair=d_pair,
                                                        n_head_msa=n_head_msa,
                                                        n_head_pair=n_head_pair,
                                                        d_hidden_msa=8,
                                                        d_hidden=d_hidden,
                                                        p_drop=p_drop,
                                                        use_global_attn=True,
                                                        SE3_param=SE3_param_full)
                                                        for i in range(n_extra_block)])

        # Update with seed sequences
        if n_main_block > 0:
            self.main_block = nn.ModuleList([IterBlock(d_msa=d_msa, d_pair=d_pair,
                                                       n_head_msa=n_head_msa,
                                                       n_head_pair=n_head_pair,
                                                       d_hidden=d_hidden,
                                                       p_drop=p_drop,
                                                       use_global_attn=False,
                                                       SE3_param=SE3_param_full)
                                                       for i in range(n_main_block)])

        # Final SE(3) refinement
        if n_ref_block > 0:
            self.str_refiner = Str2Str(d_msa=d_msa, d_pair=d_pair,
                                       d_state=SE3_param_topk['l0_out_features'],
                                       SE3_param=SE3_param_topk,
                                       nextra_l0=4*NTOTALDOFS,
                                       nextra_l1=6,
                                       p_drop=p_drop)


    def get_gradients(self, seq, idx, xyz, alpha):
        dbonddxyz, dbonddalpha = calc_BB_bond_geom_grads(
            seq, idx, xyz, alpha,
            self.xyzconverter.compute_all_atom
        )

        dljdxyz, dljdalpha = calc_lj_grads(
            seq, xyz, alpha, 
            self.xyzconverter.compute_all_atom,
            self.aamask, 
            self.ljlk_parameters, 
            self.lj_correction_parameters, 
            self.num_bonds, 
            lj_lin=self.lj_lin,
            training=self.training
        )

        extra_l1 = torch.cat((dbonddxyz[0],dljdxyz[0]), dim=1)
        extra_l0 = torch.cat((
            dbonddalpha.reshape(1,-1,2*NTOTALDOFS),
            dljdalpha.reshape(1,-1,2*NTOTALDOFS)
        ), dim=2)

        return extra_l0, extra_l1

    def forward(self, seq, msa, msa_full, pair, xyz, state, idx, same_chain, use_checkpoint=False):
        # input:
        #   msa: initial MSA embeddings (N, L, d_msa)
        #   pair: initial residue pair embeddings (L, L, d_pair)
        xyz_s = list()
        alpha_s = list()

        for i_m in range(self.n_extra_block):
            msa_full, pair, xyz, state, alpha = self.extra_block[i_m](msa_full, pair,
                                                               xyz, state, idx,
                                                               use_checkpoint=use_checkpoint)
            xyz_s.append(xyz)
            alpha_s.append(alpha)

        for i_m in range(self.n_main_block):
            msa, pair, xyz, state, alpha = self.main_block[i_m](msa, pair,
                                                         xyz, state, idx,
                                                         use_checkpoint=use_checkpoint)
            xyz_s.append(xyz)
            alpha_s.append(alpha)

        # refinement - pass gradients through _first_ and randint(2,N) iteration
        if (not xyz.requires_grad):
            ncycles = self.n_ref_block
        else:
            ncycles = np.random.randint(2,self.n_ref_block)

        for i_m in range(self.n_ref_block):
            extra_l0, extra_l1 = self.get_gradients(
                seq, idx, xyz.detach(), alpha.detach()
            )

            xyz, state, alpha = self.str_refiner(
                msa.float(), pair.float(), xyz.detach().float(), state.float(), idx, 
                extra_l0.float(), extra_l1.float(), top_k=128)

            xyz_s.append(xyz)
            alpha_s.append(alpha)

        _, xyzallatom = self.xyzconverter.compute_all_atom(seq, xyz, alpha)  # think about detach here...

        xyzs = torch.stack(xyz_s, dim=0)
        alphas = torch.stack(alpha_s, dim=0)

        return msa, pair, xyzs, alphas, xyzallatom, state