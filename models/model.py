import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from PIL import Image
import time
import cv2
import pdb

from .transformer.Beam import Beam
from .transformer.Decoder import Decoder
from .transformer.Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD
from .resnet import resnet34, extract_g

class MODEL(nn.Module):

    def __init__(self, n_bm, n_vocab, 
        inputDataType='torch.cuda.FloatTensor', maxBatch=256, dec_layer=1, LR=True):
        super(MODEL, self).__init__()
        self.device = torch.device('cuda')
        self.n_bm = n_bm
        d_model = 1024
        d_word_vec = 512
        self.max_seq_len = 36
        n_tgt_vocab = n_vocab + 4  # add BOS EOS PAD UNK
        self.encoder = resnet34()
        
        self.extract_g = extract_g(512)
        self.conv1x1 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
        self.LR = LR
        #self.src_position_enc = nn.Embedding(15, d_word_vec, padding_idx=0)

        self.decoder1 = Decoder(
            n_tgt_vocab=n_tgt_vocab, len_max_seq=self.max_seq_len,
            d_word_vec=d_word_vec, d_model=d_model, d_inner=2048,
            n_layers=dec_layer, n_head=16, d_k=64, d_v=64,
            dropout=0.1)
        if LR is True:
           self.decoder2 = Decoder(
               n_tgt_vocab=n_tgt_vocab, len_max_seq=self.max_seq_len,
               d_word_vec=d_word_vec, d_model=d_model, d_inner=2048,
               n_layers=dec_layer, n_head=16, d_k=64, d_v=64,
               dropout=0.1)
            
        self.tgt_word_prj = nn.Linear(d_model, n_tgt_vocab, bias=False)
        nn.init.xavier_normal_(self.tgt_word_prj.weight)
        self.x_logit_scale = 1.
        
      
    def beam_decode_step(self,inst_dec_beams, len_dec_seq, src_seq, enc_output, global_feat, inst_idx_to_position_map, n_bm,LorR=1):

        def prepare_beam_dec_seq(inst_dec_beams, len_dec_seq):
            dec_partial_seq = [b.get_current_state() for b in inst_dec_beams if not b.done]
            dec_partial_seq = torch.stack(dec_partial_seq).to(self.device)
            dec_partial_seq = dec_partial_seq.view(-1, len_dec_seq)
            return dec_partial_seq

        def prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm):
            dec_partial_pos = torch.arange(1, len_dec_seq + 1, dtype=torch.long, device=self.device)
            dec_partial_pos = dec_partial_pos.unsqueeze(0).repeat(n_active_inst * n_bm, 1)
            return dec_partial_pos

        def predict_word(dec_seq, dec_pos, src_seq, enc_output, global_feat, n_active_inst, n_bm, LorR=1):
            if LorR==1:
               dec_output, slf_attns, enc_attns = self.decoder1(dec_seq, dec_pos, src_seq, enc_output, global_feat, return_attns=True,if_test=True)
            elif LorR==2:
               dec_output, slf_attns, enc_attns = self.decoder2(dec_seq, dec_pos, src_seq, enc_output, global_feat, return_attns=True,if_test=True)
            else: 
               print('ERROR in predict_word')
               assert 0==1
            dec_output = dec_output[:, -1, :] 
                          
            dec_output_prj = self.tgt_word_prj(dec_output)
            
            word_prob = F.log_softmax(dec_output_prj*self.x_logit_scale, dim=1)
            word_prob = word_prob.view(n_active_inst, n_bm, -1)

            return word_prob

        def collect_active_inst_idx_list(inst_beams, word_prob, inst_idx_to_position_map): 
            
            active_inst_idx_list = []
            for inst_idx, inst_position in inst_idx_to_position_map.items():
                is_inst_complete = inst_beams[inst_idx].advance(word_prob[inst_position])
                if not is_inst_complete:
                    active_inst_idx_list += [inst_idx]
            
            return active_inst_idx_list
        
        n_active_inst = len(inst_idx_to_position_map)      
        
        dec_seq = prepare_beam_dec_seq(inst_dec_beams, len_dec_seq)
        dec_pos = prepare_beam_dec_pos(len_dec_seq, n_active_inst, n_bm)
        if len_dec_seq==1:
           for i in range(len(dec_seq)):
               dec_seq[i][0] = BOS
               
        sorted_map = sorted(inst_idx_to_position_map.items(),key = lambda x:x[1])
        active_index = [ori_indx for ori_indx,new_indx in sorted_map]
        global_feat = global_feat[active_index]
        
        word_prob = predict_word(dec_seq, dec_pos, src_seq, enc_output, global_feat, n_active_inst, n_bm, LorR=LorR)
        
        # Update the beam with predicted word prob information and collect incomplete instances
        active_inst_idx_list = collect_active_inst_idx_list(
            inst_dec_beams, word_prob, inst_idx_to_position_map)

        return active_inst_idx_list

    def collect_hypothesis_and_scores(self,inst_dec_beams, n_best):
        all_hyp, all_scores = [], []
        for inst_idx in range(len(inst_dec_beams)):
            scores, tail_idxs = inst_dec_beams[inst_idx].sort_scores()
            all_scores += [scores[:n_best]]
            hyps = [inst_dec_beams[inst_idx].get_hypothesis(i) for i in tail_idxs[:n_best]]
            all_hyp += [hyps]
        return all_hyp, all_scores
    def get_inst_idx_to_tensor_position_map(self, inst_idx_list):
        return {inst_idx: tensor_position for tensor_position, inst_idx in enumerate(inst_idx_list)}
    def collect_active_part(self, beamed_tensor, curr_active_inst_idx, n_prev_active_inst, n_bm):
        ''' Collect tensor parts associated to active instances. '''
        n_curr_active_inst = len(curr_active_inst_idx)
        if len(beamed_tensor.size())==2:
           _, d_hs = beamed_tensor.size()
           new_shape = (n_curr_active_inst * n_bm, d_hs)
        elif len(beamed_tensor.size())==3:
           _, d_hs1, d_hs2 = beamed_tensor.size()
           new_shape = (n_curr_active_inst * n_bm, d_hs1, d_hs2)
        else:
           print(beamed_tensor.size())
           assert 0==1
        beamed_tensor = beamed_tensor.view(n_prev_active_inst, -1)
        beamed_tensor = beamed_tensor.index_select(0, curr_active_inst_idx)
        beamed_tensor = beamed_tensor.view(new_shape)
        return beamed_tensor
    def collate_active_info(self,src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list, n_bm):
        # Sentences which are still active are collected,
        # so the decoder will not run on completed sentences.
        n_prev_active_inst = len(inst_idx_to_position_map)

        
        active_inst_idx = [inst_idx_to_position_map[k] for k in active_inst_idx_list]
        active_inst_idx.sort()
        active_inst_idx = torch.LongTensor(active_inst_idx).to(self.device)
        
        active_src_seq = self.collect_active_part(src_seq, active_inst_idx, n_prev_active_inst, n_bm)
        active_src_enc = self.collect_active_part(src_enc, active_inst_idx, n_prev_active_inst, n_bm)

        active_inst_idx_list.sort()
        
        #active_inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
        active_inst_idx_to_position_map = {}
        count = 0
        for idx in active_inst_idx_list:
            active_inst_idx_to_position_map[idx] = count
            count += 1
        return active_src_seq, active_src_enc, active_inst_idx_to_position_map

        
    def forward(self, x, length_ori, text1_ori, text2_ori, test=False, cpu_texts=None):
        N = x.shape[0]
        if len(text1_ori.shape)==2:
           text1_ori = text1_ori[0]
           if text2_ori is not None: 
              text2_ori = text2_ori[0]
                      
        if len(x.shape)==4 and x.shape[1]==1:
           x = x.expand(x.shape[0],3,x.shape[2],x.shape[3])
        if torch.max(length_ori) > self.max_seq_len-1:
           print(cpu_texts)
           print(length_ori)
           assert torch.max(length_ori) <= self.max_seq_len-1            
        
        # Encoder #
        cnn_feat = self.encoder(x)
        global_feat = self.extract_g(cnn_feat)
        
        text1_new = text1_ori
        text2_new = text2_ori
        length_new = length_ori                     
        
        cnn_feat = self.conv1x1(cnn_feat)
        cnn_feat = cnn_feat.squeeze(-1).permute(0, 2, 3, 1).contiguous().view(N, -1, cnn_feat.shape[1])
        
        
        # Decoding ...
        if test==True and self.n_bm >=1:
            src_enc = cnn_feat
            n_inst, len_s, d_h = src_enc.shape
            src_seq = torch.ones((n_inst,len_s)).long().cuda()
            assert n_inst==N
            src_seq = src_seq.repeat(1, self.n_bm).view(n_inst * self.n_bm, len_s)
            src_enc = src_enc.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, len_s, d_h)
            inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]
            active_inst_idx_list = list(range(n_inst))
            inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
            #-- Decode
            for len_dec_seq in range(1, self.max_seq_len):
                active_inst_idx_list = self.beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, global_feat, inst_idx_to_position_map, self.n_bm, LorR=1)
                if not active_inst_idx_list:
                    break
                src_seq, src_enc, inst_idx_to_position_map = self.collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list,self.n_bm)
            batch_hyp1, batch_scores1 = self.collect_hypothesis_and_scores(inst_dec_beams, 1)
            
            seq_stacked = []
            for i,lenth in enumerate(length_new):
                old_len = len(seq_stacked)
                lenth_add5eos = lenth + 5
                if len(batch_hyp1[i][0])>=lenth_add5eos:
                   seq_stacked.extend(batch_hyp1[i][0][0:lenth_add5eos])
                else:
                   pad_num = lenth_add5eos - len(batch_hyp1[i][0])
                   seq_stacked.extend(batch_hyp1[i][0])
                   for pad_i in range(pad_num):
                       seq_stacked.extend([EOS])
                #assert len(cpu_texts[i])+5==(len(seq_stacked)-old_len)
            seq_stacked1 = torch.Tensor(seq_stacked).long().cuda()

            if text2_ori is not None:
               src_enc = cnn_feat
               n_inst, len_s, d_h = src_enc.shape
               src_seq = torch.ones((n_inst,len_s)).long().cuda()
               assert n_inst==N
               src_seq = src_seq.repeat(1, self.n_bm).view(n_inst * self.n_bm, len_s)
               src_enc = src_enc.repeat(1, self.n_bm, 1).view(n_inst * self.n_bm, len_s, d_h)
               inst_dec_beams = [Beam(self.n_bm, device=self.device) for _ in range(n_inst)]
               active_inst_idx_list = list(range(n_inst))
               inst_idx_to_position_map = self.get_inst_idx_to_tensor_position_map(active_inst_idx_list)
               #-- Decode
               for len_dec_seq in range(1, self.max_seq_len):
                   active_inst_idx_list = self.beam_decode_step(inst_dec_beams, len_dec_seq, src_seq, src_enc, global_feat, inst_idx_to_position_map, self.n_bm, LorR=2)
                   if not active_inst_idx_list:
                       break
                   src_seq, src_enc, inst_idx_to_position_map = self.collate_active_info(src_seq, src_enc, inst_idx_to_position_map, active_inst_idx_list,self.n_bm)
               batch_hyp2, batch_scores2 = self.collect_hypothesis_and_scores(inst_dec_beams, 1)
               
               seq_stacked = []
               for i,lenth in enumerate(length_new):
                   old_len = len(seq_stacked)
                   lenth_add5eos = lenth + 5
                   if len(batch_hyp2[i][0])>=lenth_add5eos:
                      seq_stacked.extend(batch_hyp2[i][0][0:lenth_add5eos])
                   else:
                      pad_num = lenth_add5eos - len(batch_hyp2[i][0])
                      seq_stacked.extend(batch_hyp2[i][0])
                      for pad_i in range(pad_num):
                          seq_stacked.extend([EOS])
               seq_stacked2 = torch.Tensor(seq_stacked).long().cuda() 
        else:        
            tgt_seq1 = torch.ones(N,self.max_seq_len).long().cuda() * PAD
            if text2_ori is not None: 
               tgt_seq2 = torch.ones(N,self.max_seq_len).long().cuda() * PAD
            tgt_pos = torch.zeros(N,self.max_seq_len).long().cuda()
            tgt_seq1[:,0] = BOS
            if text2_ori is not None: 
               tgt_seq2[:,0] = BOS
            text_pos = 0
            for i in range(N):
                tgt_seq1[i,1:length_new[i]+1] = text1_new[text_pos:text_pos+length_new[i]]
                if text2_ori is not None: 
                   tgt_seq2[i,1:length_new[i]+1] = text2_new[text_pos:text_pos+length_new[i]]
                text_pos += length_new[i]
                        
            for i in range(N):
              for j in range(length_new[i]+1): # BOS
                  tgt_pos[i][j] = j+1
   
            
            dec_output1, slf_attns1, enc_attns1 = self.decoder1(tgt_seq1, tgt_pos, None, cnn_feat, global_feat, return_attns=True)
            if text2_ori is not None: 
               dec_output2, slf_attns2, enc_attns2 = self.decoder2(tgt_seq2, tgt_pos, None, cnn_feat, global_feat, return_attns=True)
            seq_logit1 = self.tgt_word_prj(dec_output1) * self.x_logit_scale
            if text2_ori is not None:
               seq_logit2 = self.tgt_word_prj(dec_output2) * self.x_logit_scale
            
            if test==False:
               seq_logit1 = seq_logit1.view(-1, seq_logit1.size(2))
               if text2_ori is not None:
                  seq_logit2 = seq_logit2.view(-1, seq_logit2.size(2))
               for i,lenth in enumerate(length_new):
                   if i==0:
                      seq_stacked1 = seq_logit1[0:lenth]
                      if text2_ori is not None:
                         seq_stacked2 = seq_logit2[0:lenth]
                   else:
                      seq_stacked1 = torch.cat((seq_stacked1,seq_logit1[i*self.max_seq_len:i*self.max_seq_len+lenth]),0)
                      if text2_ori is not None:
                         seq_stacked2 = torch.cat((seq_stacked2,seq_logit2[i*self.max_seq_len:i*self.max_seq_len+lenth]),0)
            else:
               _, preds1 = seq_logit1.max(-1)
               prob1s = F.softmax(seq_logit1,-1).max(-1)[0]
               prob1 = []
               for i, probs in enumerate(prob1s):
                   local_pred = preds1[i]
                   list_of_EOS_positions = torch.nonzero(torch.eq(local_pred, EOS))
                   if len(list_of_EOS_positions)>1:
                      num = list_of_EOS_positions[0] + 1 
                   else:
                      num = self.max_seq_len -1
                   local_score = 1.0
                   for j in range(num):
                      local_score *= probs[j]
                   prob1.append(local_score)
               batch_scores1 = torch.Tensor(prob1)
               
               if text2_ori is not None:
                  prob2, preds2 = seq_logit2.max(-1)
                  prob2s = F.softmax(seq_logit2,-1).max(-1)[0]
                  prob2 = []
                  for i, probs in enumerate(prob2s):
                      local_pred = preds2[i]
                      list_of_EOS_positions = torch.nonzero(torch.eq(local_pred, EOS))
                      if len(list_of_EOS_positions)>1:
                         num = list_of_EOS_positions[0] + 1 
                      else:
                         num = self.max_seq_len -1
                      local_score = 1.0
                      for j in range(num):
                         local_score *= probs[j]
                      prob2.append(local_score)
                  batch_scores2 = torch.Tensor(prob2)
               
               seq_stacked = []
               for i,lenth in enumerate(length_ori):
                   old_len = len(seq_stacked)
                   lenth_add5eos = lenth + 5
                   if lenth_add5eos <= self.max_seq_len:
                      seq_stacked.extend(preds1[i, 0:lenth_add5eos])
                   else:
                      pad_num = lenth_add5eos - self.max_seq_len
                      seq_stacked.extend(preds1[i,0:self.max_seq_len])
                      for pad_i in range(pad_num):
                          seq_stacked.extend([EOS])
               seq_stacked1 = torch.Tensor(seq_stacked).long().cuda()
   
               if text2_ori is not None:
                  seq_stacked = []
                  for i,lenth in enumerate(length_ori):
                      old_len = len(seq_stacked)
                      lenth_add5eos = lenth + 5
                      if lenth_add5eos <= self.max_seq_len:
                         seq_stacked.extend(preds2[i, 0:lenth_add5eos])
                      else:
                         pad_num = lenth_add5eos - self.max_seq_len
                         seq_stacked.extend(preds2[i,0:self.max_seq_len])
                         for pad_i in range(pad_num):
                             seq_stacked.extend([EOS])
                  seq_stacked2 = torch.Tensor(seq_stacked).long().cuda()        
        
        if text2_ori is not None:
           if test==True:
              return seq_stacked1, batch_scores1, seq_stacked2, batch_scores2
           else:
              return seq_stacked1, seq_stacked2
        else:
           if test==True:
              return seq_stacked1, batch_scores1
           else:
              return seq_stacked1
        