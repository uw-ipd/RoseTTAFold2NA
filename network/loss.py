import torch
import torch.nn as nn
import numpy as np
from torch import einsum
from chemical import aa2num
from util import rigid_from_3_points, is_nucleic
from kinematics import get_dih
from scoring import HbHybType

# Loss functions for the training
# 1. BB rmsd loss
# 2. distance loss (or 6D loss?)
# 3. bond geometry loss
# 4. predicted lddt loss

def calc_c6d_loss(logit_s, label_s, mask_2d, eps=1e-5):
    loss_s = list()
    for i in range(len(logit_s)):
        loss = nn.CrossEntropyLoss(reduction='none')(logit_s[i], label_s[...,i]) # (B, L, L)
        loss = (mask_2d*loss).sum() / (mask_2d.sum() + eps)
        loss_s.append(loss)
    loss_s = torch.stack(loss_s)
    return loss_s

def get_t(N, Ca, C, eps=1e-5):
    I,B,L=N.shape[:3]
    Rs,Ts = rigid_from_3_points(N.view(I*B,L,3), Ca.view(I*B,L,3), C.view(I*B,L,3), eps=eps)
    Rs = Rs.view(I,B,L,3,3)
    Ts = Ts.view(I,B,L,3)
    t = Ts.unsqueeze(-2) - Ts.unsqueeze(-3)
    return torch.einsum('iblkj, iblmk -> iblmj', Rs, t) # (I,B,L,L,3) 

def calc_str_loss(
    seq, pred, true, logit_pae, mask_2d, same_chain, negative=False, 
    d_intra=10.0, d_intra_na=30.0, d_inter=30.0, 
    A=10.0, gamma=0.99, eps=1e-6
):
    '''
    Calculate Backbone FAPE loss
    Input:
        - pred: predicted coordinates (I, B, L, n_atom, 3)
        - true: true coordinates (B, L, n_atom, 3)
    Output: str loss
    '''
    I = pred.shape[0]
    true = true.unsqueeze(0)
    t_tilde_ij = get_t(true[:,:,:,0], true[:,:,:,1], true[:,:,:,2])
    t_ij = get_t(pred[:,:,:,0], pred[:,:,:,1], pred[:,:,:,2])

    difference = torch.sqrt(torch.square(t_tilde_ij-t_ij).sum(dim=-1) + eps)
    eij_label = difference[-1].clone().detach()

    clamp = torch.zeros_like(difference)

    # intra vs inter
    clamp[:,same_chain==1] = d_intra
    clamp[:,same_chain==0] = d_inter

    # na vs prot
    is_NA = is_nucleic(seq)
    is_NA = is_NA[:,:,None]*is_NA[:,None,:]
    clamp[:,is_NA] = d_intra_na

    difference = torch.clamp(difference, max=clamp)
    loss = difference / A # (I, B, L, L)

    # Get a mask information (ignore missing residue + inter-chain residues)
    # for positive cases, mask = mask_2d
    # for negative cases (non-interacting pairs) mask = mask_2d*same_chain
    if negative:
        mask = mask_2d * same_chain
    else:
        mask = mask_2d
    # calculate masked loss (ignore missing regions when calculate loss)
    sub = loss[0]
    loss = (loss[:,mask.bool()]).sum(dim=-1) / (mask.sum()+eps) # (I)

    # weighting loss
    w_loss = torch.pow(torch.full((I,), gamma, device=pred.device), torch.arange(I, device=pred.device))
    w_loss = torch.flip(w_loss, (0,))
    w_loss = w_loss / w_loss.sum()

    tot_loss = (w_loss * loss).sum()
    
    # calculate pae loss
    nbin = logit_pae.shape[1]
    bin_step = 0.5
    pae_bins = torch.linspace(bin_step, bin_step*(nbin-1), nbin-1, dtype=logit_pae.dtype, device=logit_pae.device)
    true_pae_label = torch.bucketize(eij_label, pae_bins, right=True).long()
    pae_loss = torch.nn.CrossEntropyLoss(reduction='none')(
        logit_pae, true_pae_label)

    pae_loss = (pae_loss[mask.bool()]).sum() / (mask.sum() + eps)
    return tot_loss, loss.detach(), pae_loss

#resolve rotationally equivalent sidechains
def resolve_symmetry(xs, Rsnat_all, xsnat, Rsnat_all_alt, xsnat_alt, atm_mask):
    dists = torch.linalg.norm( xs[:,:,None,:] - xs[atm_mask,:][None,None,:,:], dim=-1)
    dists_nat = torch.linalg.norm( xsnat[:,:,None,:] - xsnat[atm_mask,:][None,None,:,:], dim=-1)
    dists_natalt = torch.linalg.norm( xsnat_alt[:,:,None,:] - xsnat_alt[atm_mask,:][None,None,:,:], dim=-1)

    drms_nat = torch.sum(torch.abs(dists_nat-dists),dim=(-1,-2))
    drms_natalt = torch.sum(torch.abs(dists_nat-dists_natalt), dim=(-1,-2))

    Rsnat_symm = Rsnat_all
    xs_symm = xsnat

    toflip = drms_natalt<drms_nat

    Rsnat_symm[toflip,...] = Rsnat_all_alt[toflip,...]
    xs_symm[toflip,...] = xsnat_alt[toflip,...]

    return Rsnat_symm, xs_symm

# resolve "equivalent" natives
def resolve_equiv_natives(xs, natstack, maskstack):
    if (len(natstack.shape)==4):
        return natstack, maskstack
    if (natstack.shape[1]==1):
        return natstack[:,0,...], maskstack[:,0,...]
    dx = torch.norm( xs[:,None,:,None,1,:]-xs[:,None,None,:,1,:], dim=-1)
    dnat = torch.norm( natstack[:,:,:,None,1,:]-natstack[:,:,None,:,1,:], dim=-1)
    delta = torch.sum( torch.abs(dnat-dx), dim=(-2,-1))
    return natstack[:,torch.argmin(delta),...], maskstack[:,torch.argmin(delta),...]


#torsion angle predictor loss
def torsionAngleLoss( alpha, alphanat, alphanat_alt, tors_mask, tors_planar, eps=1e-8 ):
    I = alpha.shape[0]
    lnat = torch.sqrt( torch.sum( torch.square(alpha), dim=-1 ) + eps )
    anorm = alpha / (lnat[...,None])

    l_tors_ij = torch.min(
            torch.sum(torch.square( anorm - alphanat[None] ),dim=-1),
            torch.sum(torch.square( anorm - alphanat_alt[None] ),dim=-1)
        )

    l_tors = torch.sum( l_tors_ij*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_norm = torch.sum( torch.abs(lnat-1.0)*tors_mask[None] ) / (torch.sum( tors_mask )*I + eps)
    l_planar = torch.sum( torch.abs( alpha[...,0] )*tors_planar[None] ) / (torch.sum( tors_planar )*I + eps)

    return l_tors+0.02*l_norm+0.02*l_planar

def compute_FAPE(Rs, Ts, xs, Rsnat, Tsnat, xsnat, Z=10.0, dclamp=10.0, eps=1e-4):
    xij = torch.einsum('rji,rsj->rsi', Rs, xs[None,...] - Ts[:,None,...])
    xij_t = torch.einsum('rji,rsj->rsi', Rsnat, xsnat[None,...] - Tsnat[:,None,...])

    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t), dim=-1 ) + eps )
    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean()

    return loss

# from Ivan: FAPE generalized over atom sets & frames
def compute_general_FAPE(X, Y, atom_mask, frames, frame_mask, Z=10.0, dclamp=10.0, eps=1e-4):
    # X (predicted) N x L x 27 x 3
    # Y (native)    1 x L x 27 x 3
    # atom_mask     1 x L x 27
    # frames        1 x L x 6 x 3
    # frame_mask    1 x L x 6

    N = X.shape[0]
    X_x = torch.gather(X, 2, frames[...,0:1].repeat(N,1,1,3))
    X_y = torch.gather(X, 2, frames[...,1:2].repeat(N,1,1,3))
    X_z = torch.gather(X, 2, frames[...,2:3].repeat(N,1,1,3))
    uX,tX = rigid_from_3_points(X_x, X_y, X_z)

    Y_x = torch.gather(Y, 2, frames[...,0:1].repeat(1,1,1,3))
    Y_y = torch.gather(Y, 2, frames[...,1:2].repeat(1,1,1,3))
    Y_z = torch.gather(Y, 2, frames[...,2:3].repeat(1,1,1,3))
    uY,tY = rigid_from_3_points(Y_x, Y_y, Y_z)

    xij = torch.einsum(
        'brji,brsj->brsi',
        uX[:,frame_mask[0]], X[:,atom_mask[0]][:,None,...] - X_y[:,frame_mask[0]][:,:,None,...]
    )

    xij_t = torch.einsum('rji,rsj->rsi', uY[frame_mask], Y[atom_mask][None,...] - Y_y[frame_mask][:,None,...])

    diff = torch.sqrt( torch.sum( torch.square(xij-xij_t[None,...]), dim=-1 ) + eps )
    
    loss = (1.0/Z) * (torch.clamp(diff, max=dclamp)).mean(dim=(1,2))
    return loss

def angle(a, b, c, eps=1e-6):
    '''
    Calculate cos/sin angle between ab and cb
    a,b,c have shape of (B, L, 3)
    '''
    B,L = a.shape[:2]

    u1 = a-b
    u2 = c-b

    u1_norm = torch.norm(u1, dim=-1, keepdim=True) + eps
    u2_norm = torch.norm(u2, dim=-1, keepdim=True) + eps

    # normalize u1 & u2 --> make unit vector
    u1 = u1 / u1_norm
    u2 = u2 / u2_norm
    u1 = u1.reshape(B*L, 3)
    u2 = u2.reshape(B*L, 3)

    # sin_theta = norm(a cross b)/(norm(a)*norm(b))
    # cos_theta = norm(a dot b) / (norm(a)*norm(b))
    sin_theta = torch.norm(torch.cross(u1, u2, dim=1), dim=1, keepdim=True).reshape(B, L, 1) # (B,L,1)
    cos_theta = torch.matmul(u1[:,None,:], u2[:,:,None]).reshape(B, L, 1)
    
    return torch.cat([cos_theta, sin_theta], axis=-1) # (B, L, 2)

def length(a, b):
    return torch.norm(a-b, dim=-1)

def torsion(a,b,c,d, eps=1e-6):
    #A function that takes in 4 atom coordinates:
    # a - [B,L,3]
    # b - [B,L,3]
    # c - [B,L,3]
    # d - [B,L,3]
    # and returns cos and sin of the dihedral angle between those 4 points in order a, b, c, d
    # output - [B,L,2]
    u1 = b-a
    u1 = u1 / (torch.norm(u1, dim=-1, keepdim=True) + eps)
    u2 = c-b
    u2 = u2 / (torch.norm(u2, dim=-1, keepdim=True) + eps)
    u3 = d-c
    u3 = u3 / (torch.norm(u3, dim=-1, keepdim=True) + eps)
    #
    t1 = torch.cross(u1, u2, dim=-1) #[B, L, 3]
    t2 = torch.cross(u2, u3, dim=-1)
    t1_norm = torch.norm(t1, dim=-1, keepdim=True)
    t2_norm = torch.norm(t2, dim=-1, keepdim=True)
    
    cos_angle = torch.matmul(t1[:,:,None,:], t2[:,:,:,None])[:,:,0]
    sin_angle = torch.norm(u2, dim=-1,keepdim=True)*(torch.matmul(u1[:,:,None,:], t2[:,:,:,None])[:,:,0])
    
    cos_sin = torch.cat([cos_angle, sin_angle], axis=-1)/(t1_norm*t2_norm+eps) #[B,L,2]
    return cos_sin

# ideal N-C distance and cos(angles)
# for NA, use P-O dist and cos(angles)
def calc_BB_bond_geom(
    seq, idx, pred, 
    ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, 
    ideal_OP=1.607, ideal_OPO=-0.3106, ideal_OPC=-0.4970, 
    sig_len=0.02, sig_ang=0.05,  eps=1e-6):
    '''
    Calculate backbone bond geometry (bond length and angle) and put loss on them
    Input:
     - pred: predicted coords (B, L, :, 3), 0; N / 1; CA / 2; C
     - true: True coords (B, L, :, 3)
    Output:
     - bond length loss, bond angle loss
    '''
    def cosangle( A,B,C ):
        #print (torch.isnan(A).sum(),torch.isnan(B).sum(),torch.isnan(C).sum())
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    B, L = pred.shape[:2]

    bonded = (idx[:,1:] - idx[:,:-1])==1
    is_NA = is_nucleic(seq)[:-1]
    is_prot = ~is_NA

    # bond length: C-N
    blen_CN_pred  = length(pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1) # (B, L-1)
    CN_loss = torch.clamp( torch.abs(blen_CN_pred - ideal_NC) - sig_len, min=0.0 )
    CN_loss = (bonded*is_prot*CN_loss).sum() / ((bonded*is_prot).sum() + eps)

    # bond length: P-O
    blen_OP_pred  = length(pred[:,:-1,8], pred[:,1:,1]).reshape(B,L-1) # (B, L-1)
    OP_loss = torch.clamp( torch.abs(blen_OP_pred - ideal_OP) - sig_len, min=0.0 )
    OP_loss = (bonded*is_NA*OP_loss).sum() / ((bonded*is_NA).sum() + eps)

    blen_loss = CN_loss + OP_loss

    # bond angle: CA-C-N, C-N-CA
    bang_CACN_pred = cosangle(pred[:,:-1,1], pred[:,:-1,2], pred[:,1:,0]).reshape(B,L-1)
    bang_CNCA_pred = cosangle(pred[:,:-1,2], pred[:,1:,0], pred[:,1:,1]).reshape(B,L-1)
    CACN_loss = torch.clamp( torch.abs(bang_CACN_pred - ideal_CACN) - sig_ang,  min=0.0 )
    CACN_loss = (bonded*is_prot*CACN_loss).sum() / ((bonded*is_prot).sum() + eps)
    CNCA_loss = torch.clamp( torch.abs(bang_CNCA_pred - ideal_CNCA) - sig_ang,  min=0.0 )
    CNCA_loss = (bonded*is_prot*CNCA_loss).sum() / ((bonded*is_prot).sum() + eps)

    # bond angle: O1PO, O2PO, OPC
    bang_O1PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,1], pred[:,1:,0]).reshape(B,L-1)
    bang_O2PO_pred = cosangle(pred[:,:-1,8], pred[:,1:,1], pred[:,1:,2]).reshape(B,L-1)
    bang_OPC_pred  = cosangle(pred[:,:-1,7], pred[:,:-1,8], pred[:,1:,1]).reshape(B,L-1)
    O1PO_loss = torch.clamp( torch.abs(bang_O1PO_pred - ideal_OPO) - sig_ang,  min=0.0 )
    O1PO_loss = (bonded*is_NA*O1PO_loss).sum() / ((bonded*is_NA).sum() + eps)
    O2PO_loss = torch.clamp( torch.abs(bang_O2PO_pred - ideal_OPO) - sig_ang,  min=0.0 )
    O2PO_loss = (bonded*is_NA*O2PO_loss).sum() / ((bonded*is_NA).sum() + eps)
    OPC_loss = torch.clamp( torch.abs(bang_OPC_pred - ideal_OPC) - sig_ang,  min=0.0 )
    OPC_loss = (bonded*is_NA*OPC_loss).sum() / ((bonded*is_NA).sum() + eps)

    bang_loss = CACN_loss + CNCA_loss + O1PO_loss + O2PO_loss + OPC_loss

    return blen_loss+bang_loss


# LJ loss
#  custom backwards for mem efficiency
class LJLoss(torch.autograd.Function):
    @staticmethod
    def ljVdV(deltas, sigma, epsilon, lj_lin, eps):
        dist = torch.sqrt( torch.sum ( torch.square( deltas ), dim=-1 ) + eps )
        linpart = dist<lj_lin*sigma
        deff = dist.clone()
        deff[linpart] = lj_lin*sigma[linpart]
        sd = sigma / deff
        sd2 = sd*sd
        sd6 = sd2 * sd2 * sd2
        sd12 = sd6 * sd6
        ljE = epsilon * (sd12 - 2 * sd6)
        ljE[linpart] += epsilon[linpart] * (
            -12 * sd12[linpart]/deff[linpart] + 12 * sd6[linpart]/deff[linpart]
        ) * (dist[linpart]-deff[linpart])

        # works for linpart too
        dljEdd_over_r = epsilon * (-12 * sd12/deff + 12 * sd6/deff) / (dist)

        return ljE.sum(), dljEdd_over_r

    @staticmethod
    def forward(
        ctx, xs, seq, aamask, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training
    ):
        L,A = xs.shape[:2]

        ds_res = torch.sqrt( torch.sum ( torch.square( 
            xs.detach()[:,None,1,:]-xs.detach()[None,:,1,:]), dim=-1 ))
        rs = torch.triu_indices(L,L,0, device=xs.device)
        ri,rj = rs[0],rs[1]

        # batch during inference for huge systems
        BATCHSIZE = len(ri)
        if (not training):
            BATCHSIZE = 65536

        #print (BATCHSIZE, (len(ri)-1)//BATCHSIZE + 1)

        ljval = 0
        dljEdx = torch.zeros_like(xs, dtype=torch.float)

        for i_batch in range((len(ri)-1)//BATCHSIZE + 1):
            idx = torch.arange(
                i_batch*BATCHSIZE, 
                min( (i_batch+1)*BATCHSIZE, len(ri)),
                device=xs.device
            )
            rii,rjj = ri[idx],rj[idx] 

            ridx,ai,aj = (
                aamask[seq[rii]][:,:,None]*aamask[seq[rjj]][:,None,:]
            ).nonzero(as_tuple=True)
            deltas = xs[rii,:,None,:]-xs[rjj,None,:,:]
            seqi,seqj = seq[rii[ridx]], seq[rjj[ridx]]

            mask = torch.ones_like(ridx, dtype=torch.bool) # are atoms defined?

            intrares = (rii[ridx]==rjj[ridx])
            mask[intrares*(ai<aj)] = False  # upper tri (atoms)

            # count-pair
            mask[intrares] *= num_bonds[seqi[intrares],ai[intrares],aj[intrares]]>=4
            pepbondres = ri[ridx]+1==rj[ridx]
            mask[pepbondres] *= (
                num_bonds[seqi[pepbondres],ai[pepbondres],2]
                + num_bonds[seqj[pepbondres],0,aj[pepbondres]]
                + 1) >=4

            # apply mask.  only interactions to be scored remain
            ai,aj,seqi,seqj,ridx = ai[mask],aj[mask],seqi[mask],seqj[mask],ridx[mask]
            deltas = deltas[ridx,ai,aj]

            # hbond correction
            use_hb_dis = (
                ljcorr[seqi,ai,0]*ljcorr[seqj,aj,1] 
                + ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] ).nonzero()
            use_ohdon_dis = ( # OH are both donors & acceptors
                ljcorr[seqi,ai,0]*ljcorr[seqi,ai,1]*ljcorr[seqj,aj,0] 
                +ljcorr[seqi,ai,0]*ljcorr[seqj,aj,0]*ljcorr[seqj,aj,1] 
            ).nonzero()
            use_hb_hdis = (
                ljcorr[seqi,ai,2]*ljcorr[seqj,aj,1] 
                +ljcorr[seqi,ai,1]*ljcorr[seqj,aj,2] 
            ).nonzero()

            # disulfide correction
            potential_disulf = (ljcorr[seqi,ai,3]*ljcorr[seqj,aj,3] ).nonzero()

            ljrs = ljparams[seqi,ai,0] + ljparams[seqj,aj,0]
            ljrs[use_hb_dis] = lj_hb_dis
            ljrs[use_ohdon_dis] = lj_OHdon_dis
            ljrs[use_hb_hdis] = lj_hbond_hdis

            ljss = torch.sqrt( ljparams[seqi,ai,1] * ljparams[seqj,aj,1] + eps )
            ljss [potential_disulf] = 0.0

            natoms = torch.sum(aamask[seq])
            ljval_i,dljEdd_i = LJLoss.ljVdV(deltas,ljrs,ljss,lj_lin,eps)

            ljval += ljval_i / natoms

            # sum per-atom-pair grads into per-atom grads
            # note this is stochastic op on GPU
            idxI,idxJ = rii[ridx]*A + ai, rjj[ridx]*A + aj
            dljEdx.view(-1,3).index_add_(0, idxI, dljEdd_i[:,None]*deltas, alpha=1.0/natoms)
            dljEdx.view(-1,3).index_add_(0, idxJ, dljEdd_i[:,None]*deltas, alpha=-1.0/natoms)

        ctx.save_for_backward(dljEdx)

        return ljval

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        dljEdx, = ctx.saved_tensors
        return (
            grad_output * dljEdx, 
            None, None, None, None, None, None, None, None, None, None, None
        )


# Rosetta-like version of LJ (fa_atr+fa_rep)
#   lj_lin is switch from linear to 12-6.  Smaller values more sharply penalize clashes
def calc_lj(
    seq, xs, aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    lj = LJLoss.apply
    ljval = lj(
        xs, seq, aamask, ljparams, ljcorr, num_bonds, 
        lj_lin, lj_hb_dis, lj_OHdon_dis, lj_hbond_hdis, eps, training)

    return ljval


def calc_hb(
    seq, xs, aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8
):
    def evalpoly( ds, xrange, yrange, coeffs ):
        v = coeffs[...,0]
        for i in range(1,10):
            v = v * ds + coeffs[...,i]
        minmask = ds<xrange[...,0]
        v[minmask] = yrange[minmask][...,0]
        maxmask = ds>xrange[...,1]
        v[maxmask] = yrange[maxmask][...,1]
        return v

    def cosangle( A,B,C ):
        AB = A-B
        BC = C-B
        ABn = torch.sqrt( torch.sum(torch.square(AB),dim=-1) + eps)
        BCn = torch.sqrt( torch.sum(torch.square(BC),dim=-1) + eps)
        return torch.clamp(torch.sum(AB*BC,dim=-1)/(ABn*BCn), -0.999,0.999)

    hbts = hbtypes[seq]
    hbba = hbbaseatoms[seq]

    rh,ah = (hbts[...,0]>=0).nonzero(as_tuple=True)
    ra,aa = (hbts[...,1]>=0).nonzero(as_tuple=True)
    D_xs = xs[rh,hbba[rh,ah,0]][:,None,:]
    H_xs = xs[rh,ah][:,None,:]
    A_xs = xs[ra,aa][None,:,:]
    B_xs = xs[ra,hbba[ra,aa,0]][None,:,:]
    B0_xs = xs[ra,hbba[ra,aa,1]][None,:,:]
    hyb = hbts[ra,aa,2]
    polys = hbpolys[hbts[rh,ah,0][:,None],hbts[ra,aa,1][None,:]]

    AH = torch.sqrt( torch.sum( torch.square( H_xs-A_xs), axis=-1) + eps )
    AHD = torch.acos( cosangle( B_xs, A_xs, H_xs) )
    
    Es = polys[...,0,0]*evalpoly(
        AH,polys[...,0,1:3],polys[...,0,3:5],polys[...,0,5:])
    Es += polys[...,1,0] * evalpoly(
        AHD,polys[...,1,1:3],polys[...,1,3:5],polys[...,1,5:])

    Bm = 0.5*(B0_xs[:,hyb==HbHybType.RING]+B_xs[:,hyb==HbHybType.RING])
    cosBAH = cosangle( Bm, A_xs[:,hyb==HbHybType.RING], H_xs )
    Es[:,hyb==HbHybType.RING] += polys[:,hyb==HbHybType.RING,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.RING,2,1:3], 
        polys[:,hyb==HbHybType.RING,2,3:5], 
        polys[:,hyb==HbHybType.RING,2,5:])

    cosBAH1 = cosangle( B_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    cosBAH2 = cosangle( B0_xs[:,hyb==HbHybType.SP3], A_xs[:,hyb==HbHybType.SP3], H_xs )
    Esp3_1 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH1, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Esp3_2 = polys[:,hyb==HbHybType.SP3,2,0] * evalpoly(
        cosBAH2, 
        polys[:,hyb==HbHybType.SP3,2,1:3], 
        polys[:,hyb==HbHybType.SP3,2,3:5], 
        polys[:,hyb==HbHybType.SP3,2,5:])
    Es[:,hyb==HbHybType.SP3] += torch.log(
        torch.exp(Esp3_1 * hb_sp3_softmax_fade)
        + torch.exp(Esp3_2 * hb_sp3_softmax_fade)
    ) / hb_sp3_softmax_fade

    cosBAH = cosangle( B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs )
    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * evalpoly(
        cosBAH, 
        polys[:,hyb==HbHybType.SP2,2,1:3], 
        polys[:,hyb==HbHybType.SP2,2,3:5], 
        polys[:,hyb==HbHybType.SP2,2,5:])

    BAH = torch.acos( cosBAH )
    B0BAH = get_dih(B0_xs[:,hyb==HbHybType.SP2], B_xs[:,hyb==HbHybType.SP2], A_xs[:,hyb==HbHybType.SP2], H_xs)

    d,m,l = hb_sp2_BAH180_rise, hb_sp2_range_span, hb_sp2_outer_width
    Echi = torch.full_like( B0BAH, m-0.5 )

    mask1 = BAH>np.pi * 2.0 / 3.0
    H = 0.5 * (torch.cos(2 * B0BAH) + 1)
    F = d / 2 * torch.cos(3 * (np.pi - BAH[mask1])) + d / 2 - 0.5
    Echi[mask1] = H[mask1] * F + (1 - H[mask1]) * d - 0.5

    mask2 = BAH>np.pi * (2.0 / 3.0 - l)
    mask2 *= ~mask1
    outer_rise = torch.cos(np.pi - (np.pi * 2 / 3 - BAH[mask2]) / l)
    F = m / 2 * outer_rise + m / 2 - 0.5
    G = (m - d) / 2 * outer_rise + (m - d) / 2 + d - 0.5
    Echi[mask2] = H[mask2] * F + (1 - H[mask2]) * d - 0.5

    Es[:,hyb==HbHybType.SP2] += polys[:,hyb==HbHybType.SP2,2,0] * Echi

    tosquish = torch.logical_and(Es > -0.1,Es < 0.1)
    Es[tosquish] = -0.025 + 0.5 * Es[tosquish] - 2.5 * torch.square(Es[tosquish])
    Es[Es > 0.1] = 0.
    return (torch.sum( Es ) / torch.sum(aamask[seq]))

@torch.enable_grad()
def calc_BB_bond_geom_grads(
    seq, idx, xyz, alpha, toaa, 
    ideal_NC=1.329, ideal_CACN=-0.4415, ideal_CNCA=-0.5255, 
    ideal_OP=1.607, ideal_POP=-0.3106, ideal_OPC=-0.4970, 
    sig_len=0.02, sig_ang=0.05, eps=1e-8
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Ebond = calc_BB_bond_geom(
        seq[0], idx, xyzaa,
        ideal_NC, ideal_CACN, ideal_CNCA, 
        ideal_OP, ideal_POP, ideal_OPC, 
        sig_len, sig_ang, eps
    )
    return torch.autograd.grad(Ebond, (xyz,alpha))


@torch.enable_grad()
def calc_lj_grads(
    seq, xyz, alpha, toaa, 
    aamask, ljparams, ljcorr, num_bonds, 
    lj_lin=0.85, lj_hb_dis=3.0, lj_OHdon_dis=2.6, lj_hbond_hdis=1.75, 
    lj_maxrad=-1.0, eps=1e-8,
    training=True
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Elj = calc_lj(
        seq[0], 
        xyzaa[0], 
        aamask, 
        ljparams, 
        ljcorr, 
        num_bonds, 
        lj_lin, 
        lj_hb_dis, 
        lj_OHdon_dis, 
        lj_hbond_hdis, 
        lj_maxrad,
        eps,
        training
    )
    return torch.autograd.grad(Elj, (xyz,alpha))

@torch.enable_grad()
def calc_hb_grads(
    seq, xyz, alpha, toaa, 
    aamask, hbtypes, hbbaseatoms, hbpolys,
    hb_sp2_range_span=1.6, hb_sp2_BAH180_rise=0.75, hb_sp2_outer_width=0.357, 
    hb_sp3_softmax_fade=2.5, threshold_distance=6.0, eps=1e-8, normalize=True
):
    xyz.requires_grad_(True)
    alpha.requires_grad_(True)
    _, xyzaa = toaa(seq, xyz, alpha)
    Ehb = calc_hb(
        seq[0], 
        xyzaa[0], 
        aamask, 
        hbtypes, 
        hbbaseatoms, 
        hbpolys,
        hb_sp2_range_span,
        hb_sp2_BAH180_rise,
        hb_sp2_outer_width, 
        hb_sp3_softmax_fade,    
        threshold_distance,
        eps)
    return torch.autograd.grad(Ehb, (xyz,alpha))

def calc_lddt(pred_ca, true_ca, mask_crds, mask_2d, same_chain, negative=False, interface=False, eps=1e-6):
    # Input
    # pred_ca: predicted CA coordinates (I, B, L, 3)
    # true_ca: true CA coordinates (B, L, 3)
    # pred_lddt: predicted lddt values (I-1, B, L)

    I, B, L = pred_ca.shape[:3]
    
    
    pred_ca = pred_ca.contiguous()
    true_ca = true_ca.contiguous()

    pred_dist = torch.cdist(pred_ca, pred_ca) # (I, B, L, L)
    true_dist = torch.cdist(true_ca, true_ca).unsqueeze(0) # (1, B, L, L)

    mask = torch.logical_and(true_dist > 0.0, true_dist < 15.0) # (1, B, L, L)
    # update mask information
    mask *= mask_2d[None]
    if negative:
        mask *= same_chain.bool()[None]
    elif interface:
        # ignore atoms between the same chain
        mask *= ~same_chain.bool()[None]

    mask_crds = mask_crds * (mask[0].sum(dim=-1) != 0)

    delta = torch.abs(pred_dist-true_dist) # (I, B, L, L)

    true_lddt = torch.zeros((I,B,L), device=pred_ca.device)
    for distbin in [0.5, 1.0, 2.0, 4.0]:
        true_lddt += 0.25*torch.sum((delta<=distbin)*mask, dim=-1) / (torch.sum(mask, dim=-1) + eps)
    
    true_lddt = mask_crds*true_lddt
    true_lddt = true_lddt.sum(dim=(1,2)) / (mask_crds.sum() + eps)
    return true_lddt


#fd allatom lddt
def calc_allatom_lddt(P, Q, idx, atm_mask, eps=1e-6):
    # P - N x L x 27 x 3
    # Q - L x 27 x 3
    N, L = P.shape[:2]

    # distance matrix
    Pij = torch.square(P[:,:,None,:,None,:]-P[:,None,:,None,:,:]) # (N, L, L, 27, 27)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[None,:,None,:,None,:]-Q[None,None,:,None,:,:]) # (1, L, L, 27, 27)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,:,None,:,None] * atm_mask[:,None,:,None,:]).float()

    # ignore atoms within same residue
    pair_mask *= (idx[:,:,None,None,None] != idx[:,None,:,None,None]).float() # (1, L, L, 27, 27)

    delta_PQ = torch.abs(Pij-Qij+eps) # (N, L, L, 14, 14)

    lddt = torch.zeros( (N,L,27), device=P.device ) # (N, L, 27)
    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(2,4)
            ) / ( torch.sum( pair_mask, dim=(2,4) ) + 1e-8)

    lddt = (lddt * atm_mask).sum(dim=(1,2)) / (atm_mask.sum() + eps)
    return lddt


def calc_allatom_lddt_loss(P, Q, pred_lddt, idx, atm_mask, mask_2d, same_chain, negative=False, interface=False, eps=1e-6):
    # P - N x L x 27 x 3
    # Q - L x 27 x 3
    # pred_lddt - 1 x nbucket x L
    N, L, Natm = P.shape[:3]

    # distance matrix
    Pij = torch.square(P[:,:,None,:,None,:]-P[:,None,:,None,:,:]) # (N, L, L, 27, 27)
    Pij = torch.sqrt( Pij.sum(dim=-1) + eps)
    Qij = torch.square(Q[None,:,None,:,None,:]-Q[None,None,:,None,:,:]) # (1, L, L, 27, 27)
    Qij = torch.sqrt( Qij.sum(dim=-1) + eps)

    # get valid pairs
    pair_mask = torch.logical_and(Qij>0,Qij<15).float() # only consider atom pairs within 15A
    # ignore missing atoms
    pair_mask *= (atm_mask[:,:,None,:,None] * atm_mask[:,None,:,None,:]).float()

    # ignore atoms within same residue
    pair_mask *= (idx[:,:,None,None,None] != idx[:,None,:,None,None]).float() # (1, L, L, 27, 27)
    if negative:
        # ignore atoms between different chains
        pair_mask *= same_chain.bool()[:,:,:,None,None]

    delta_PQ = torch.abs(Pij-Qij+eps) # (N, L, L, 14, 14)

    lddt = torch.zeros( (N,L,Natm), device=P.device ) # (N, L, 27)
    for distbin in (0.5,1.0,2.0,4.0):
        lddt += 0.25 * torch.sum( (delta_PQ<=distbin)*pair_mask, dim=(2,4)
            ) / ( torch.sum( pair_mask, dim=(2,4) ) + eps)

    final_lddt_by_res = torch.clamp(
        (lddt[-1]*atm_mask[0]).sum(-1)
        / (atm_mask.sum(-1) + eps), min=0.0, max=1.0)

    # calculate lddt prediction loss
    nbin = pred_lddt.shape[1]
    bin_step = 1.0 / nbin
    lddt_bins = torch.linspace(bin_step, 1.0, nbin, dtype=pred_lddt.dtype, device=pred_lddt.device)
    true_lddt_label = torch.bucketize(final_lddt_by_res[None,...], lddt_bins).long()
    lddt_loss = torch.nn.CrossEntropyLoss(reduction='none')(
        pred_lddt, true_lddt_label[-1])

    res_mask = atm_mask.any(dim=-1)
    lddt_loss = (lddt_loss * res_mask).sum() / (res_mask.sum() + eps)
   
    # method 1: average per-residue
    #lddt = lddt.sum(dim=-1) / (atm_mask.sum(dim=-1)+1e-8) # L
    #lddt = (res_mask*lddt).sum() / (res_mask.sum() + 1e-8)

    # method 2: average per-atom
    atm_mask = atm_mask * (pair_mask.sum(dim=(1,3)) != 0)
    lddt = (lddt * atm_mask).sum(dim=(1,2)) / (atm_mask.sum() + eps)

    return lddt_loss, lddt


