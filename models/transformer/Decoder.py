''' Define the Transformer model '''
import torch
import torch.nn as nn
import numpy as np
import pdb
from .Layers import DecoderLayer
from .Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD


def get_non_pad_mask(seq):
    assert seq.dim() == 2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)

def get_sinusoid_encoding_table(n_position, d_hid, padding_idx=None):
    ''' Sinusoid position encoding table '''

    def cal_angle(position, hid_idx):
        return float(position) / np.power(10000, 2 * (hid_idx // 2) / float(d_hid))

    def get_posi_angle_vec(position):
        return [cal_angle(position, hid_j) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_posi_angle_vec(pos_i) for pos_i in range(n_position)])

    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx is not None:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.

    return torch.FloatTensor(sinusoid_table)

def get_attn_key_pad_mask(seq_k, seq_q):
    ''' For masking out the padding part of key sequence. '''

    # Expand to fit the shape of key query attention matrix.
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(PAD)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)  # b x lq x lk

    return padding_mask

def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''

    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)  # b x ls x ls

    return subsequent_mask


class Decoder(nn.Module):
    ''' A decoder model with self attention mechanism. '''

    def __init__(
            self,
            n_tgt_vocab, len_max_seq, d_word_vec,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
            
        super(Decoder, self).__init__()

        n_position = len_max_seq + 1

        self.tgt_word_emb = nn.Embedding(n_tgt_vocab, d_word_vec, padding_idx=PAD)
        
        self.position_enc = nn.Embedding.from_pretrained(get_sinusoid_encoding_table(n_position, d_word_vec, padding_idx=0),freeze=True)
        #self.position_enc_stack = nn.ModuleList([nn.Embedding(n_position, d_word_vec, padding_idx=0) for _ in range(n_layers)])
        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])
        

    def forward(self, tgt_seq, tgt_pos, src_seq, enc_output, global_feat, return_attns=False, if_test=False):
        if if_test==False:
           N,L,C = enc_output.shape
           src_seq = torch.ones((N,L)).long().cuda()
           
        dec_slf_attn_list, dec_enc_attn_list = [], []

        # -- Prepare masks
        non_pad_mask = get_non_pad_mask(tgt_seq)
        dec_enc_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=tgt_seq)
        slf_attn_mask_subseq = get_subsequent_mask(tgt_seq)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=tgt_seq, seq_q=tgt_seq)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)
        
        
        # -- Forward
        
        dec_output = self.tgt_word_emb(tgt_seq) + self.position_enc(tgt_pos)
           
        if dec_output.shape[0]==global_feat.shape[0]:
           N,L,C = dec_output.shape  # batch,100,512
           global_feat_exp = global_feat.unsqueeze(1).expand(N,L,C)
        else:
           N1,L1,C1 = dec_output.shape
           N2,C2 = global_feat.shape
           assert C1==C2
           beamSize = N1/N2
           global_feat_exp = global_feat.unsqueeze(1).expand(N2,beamSize*L1,C2)
           global_feat_exp = global_feat_exp.reshape(-1,L1,C2)
        
        
        dec_output = torch.cat((global_feat_exp, dec_output),2)

        for index, dec_layer in enumerate(self.layer_stack):
            dec_output, dec_slf_attn, dec_enc_attn = dec_layer(
                dec_output, enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask,
                dec_enc_attn_mask=dec_enc_attn_mask,
                layer_idx=index)
            if return_attns:
                dec_slf_attn_list += [dec_slf_attn]
                dec_enc_attn_list += [dec_enc_attn]
                
        if return_attns:
            return dec_output, dec_slf_attn_list, dec_enc_attn_list
        return dec_output
