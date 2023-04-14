import torch
import torch.nn as nn
from chemical import NAATOKENS

class DistanceNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(DistanceNetwork, self).__init__()
        #
        self.proj_symm = nn.Linear(n_feat, 37*2)
        self.proj_asymm = nn.Linear(n_feat, 37+19)
    
        self.reset_parameter()
    
    def reset_parameter(self):
        # initialize linear layer for final logit prediction
        nn.init.zeros_(self.proj_symm.weight)
        nn.init.zeros_(self.proj_asymm.weight)
        nn.init.zeros_(self.proj_symm.bias)
        nn.init.zeros_(self.proj_asymm.bias)

    def forward(self, x):
        # input: pair info (B, L, L, C)

        # predict theta, phi (non-symmetric)
        logits_asymm = self.proj_asymm(x)
        logits_theta = logits_asymm[:,:,:,:37].permute(0,3,1,2)
        logits_phi = logits_asymm[:,:,:,37:].permute(0,3,1,2)

        # predict dist, omega
        logits_symm = self.proj_symm(x)
        logits_symm = logits_symm + logits_symm.permute(0,2,1,3)
        logits_dist = logits_symm[:,:,:,:37].permute(0,3,1,2)
        logits_omega = logits_symm[:,:,:,37:].permute(0,3,1,2)

        return logits_dist, logits_omega, logits_theta, logits_phi

class MaskedTokenNetwork(nn.Module):
    def __init__(self, n_feat, p_drop=0.1):
        super(MaskedTokenNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, NAATOKENS)
        
        self.reset_parameter()
    
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, N, L = x.shape[:3]
        logits = self.proj(x).permute(0,3,1,2).reshape(B, -1, N*L)

        return logits

class LDDTNetwork(nn.Module):
    def __init__(self, n_feat, n_bin_lddt=50):
        super(LDDTNetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_lddt)

        self.reset_parameter()

    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        logits = self.proj(x) # (B, L, 50)

        return logits.permute(0,2,1)

class PAENetwork(nn.Module):
    def __init__(self, n_feat, n_bin_pae=64):
        super(PAENetwork, self).__init__()
        self.proj = nn.Linear(n_feat, n_bin_pae)
        self.reset_parameter()
    def reset_parameter(self):
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, pair, state):
        L = pair.shape[1]
        left = state.unsqueeze(2).expand(-1,-1,L,-1)
        right = state.unsqueeze(1).expand(-1,L,-1,-1)

        logits = self.proj( torch.cat((pair, left, right), dim=-1) ) # (B, L, L, 64)

        return logits.permute(0,3,1,2)

class BinderNetwork(nn.Module):
    def __init__(self, n_hidden=64, n_bin_pae=64):
        super(BinderNetwork, self).__init__()
        #self.proj = nn.Linear(n_bin_pae, n_hidden)
        #self.classify = torch.nn.Linear(2*n_hidden, 1)
        self.classify = torch.nn.Linear(n_bin_pae, 1)
        self.reset_parameter()

    def reset_parameter(self):
        #nn.init.zeros_(self.proj.weight)
        #nn.init.zeros_(self.proj.bias)
        nn.init.zeros_(self.classify.weight)
        nn.init.zeros_(self.classify.bias)

    def forward(self, pae, same_chain):
        #logits = self.proj( pae.permute(0,2,3,1) )
        logits = pae.permute(0,2,3,1)
        #logits_intra = torch.mean( logits[same_chain==1], dim=0 )
        logits_inter = torch.mean( logits[same_chain==0], dim=0 ).nan_to_num() # all zeros if single chain
        #prob = torch.sigmoid( self.classify( torch.cat((logits_intra,logits_inter)) ) )
        prob = torch.sigmoid( self.classify( logits_inter ) )
        return prob
