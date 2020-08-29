import torch
import torch.nn as nn
from torch.autograd import Variable
import collections
import sys
sys.path.append(sys.path[0]+'/models/transformer')
from Constants import UNK,PAD,BOS,EOS,PAD_WORD,UNK_WORD,BOS_WORD,EOS_WORD
import pdb

def adjust_lr_exp(optimizer, base_lr, iter, total_iter):
  
  if iter < total_iter*1/3:
     local_lr = base_lr
  elif iter < total_iter*2/3:
     local_lr = base_lr * (0.1**1)
  else:
     local_lr = base_lr * (0.1**2)
     
  if local_lr < 0.001:
     local_lr = 0.001
     
  if optimizer.param_groups[0]['lr']!=local_lr:
       print('=============> lr adjusted to ',local_lr)
  for g in optimizer.param_groups:
    g['lr'] = local_lr
    
def str2bool(v):
  return v.lower() in ("yes", "true", "t", "1")

def addEOS(text_ori, LENGTH_ori, predict_numbers, ADD_EOS_NUM, max_len=36):
    max_len -= 1   # max_seq is 24, add BOS and EOS is 26.
    
    # LENGTH_ori include EOS, bug predict number didn't include EOS, So added 1.    
    N = len(LENGTH_ori)
    predict_numbers = predict_numbers.reshape(-1).data
    LENGTH_new = torch.clamp((torch.ceil(predict_numbers) +ADD_EOS_NUM+1), ADD_EOS_NUM+1, max_len).int()
    newtext_length = torch.sum(LENGTH_new)
    
    text_new = torch.ones(int(newtext_length.data.cpu().numpy())).long().cuda() *PAD
    old_pos = 0
    new_pos = 0
    success = 0
    for i, length in enumerate(LENGTH_ori):
       new_length = LENGTH_new[i]
       
       if new_length >= length:
          text_new[new_pos:new_pos+length] = text_ori[old_pos:old_pos+length]
          old_pos += length
          new_pos += length
          
          if new_length-length>=1:
             EOS_NUM = new_length-length
             text_new[new_pos:new_pos+EOS_NUM] = EOS
             new_pos += EOS_NUM
          success += 1
       else:
          text_new[new_pos:new_pos+new_length-1] = text_ori[old_pos:old_pos+new_length-1]
          text_new[new_pos+new_length-1] = EOS
          old_pos += length
          new_pos += new_length
    return text_new, LENGTH_new, float(success)/N
    
    
class strLabelConverterForAttention(object):
    """Convert between str and label.

    NOTE:
        Insert `EOS` to the alphabet for attention.

    Args:
        alphabet (str): set of the possible characters.
        ignore_case (bool, default=True): whether or not to ignore all of the case.
    """

    def __init__(self, alphabet, sep):
        self._scanned_list = False
        self._out_of_list = ''
        self._ignore_case = True
        self.sep = sep
        self.alphabet = alphabet.split(sep)

        self.dict = {}
        for i, item in enumerate(self.alphabet):
            # NOTE: 0 is reserved for 'blank' required by wrap_ctc
            self.dict[item] = i

    def scan(self, text):
        # print(text)
        text_tmp = text
        text = []
        for i in range(len(text_tmp)):
            text_result = ''
            for j in range(len(text_tmp[i])):
                chara = text_tmp[i][j].lower() if self._ignore_case else text_tmp[i][j]
                if chara not in self.alphabet:
                    if chara in self._out_of_list:
                        continue
                    else:
                        self._out_of_list += chara
                        file_out_of_list = open("out_of_list.txt", "a+")
                        file_out_of_list.write(chara + "\n")
                        file_out_of_list.close()
                        print('" %s " is not in alphabet...' % chara)
                        continue
                else:
                    text_result += chara
            text.append(text_result)
        text_result = tuple(text)
        self._scanned_list = True
        return text_result

    def encode(self, text, scanned=True):
        """Support batch or single str.

        Args:
            text (str or list of str): texts to convert.

        Returns:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.
        """
        self._scanned_list = scanned
                
        if not self._scanned_list:
            text = self.scan(text)

        if isinstance(text, str):
            text = [
                #self.dict[char.lower() if self._ignore_case else char]
                EOS if char==' ' else self.dict[char]
                for char in text
            ]
            length = [len(text)]

        elif isinstance(text, collections.Iterable):
            length = [len(s) for s in text]
            text = ''.join(text)
            text, _ = self.encode(text)
        return (torch.LongTensor(text), torch.LongTensor(length))

    def decode(self, t, length):
        """Decode encoded texts back into strs.

        Args:
            torch.IntTensor [length_0 + length_1 + ... length_{n - 1}]: encoded texts.
            torch.IntTensor [n]: length of each text.

        Raises:
            AssertionError: when the texts and its length does not match.

        Returns:
            text (str or list of str): texts to convert.
        """
        if length.numel() == 1:
            length = length[0]
            assert t.numel() == length, "text with length: {} does not match declared length: {}".format(t.numel(), length)
            return ''.join([' ' if i>=BOS else self.alphabet[i] for i in t])
        else:
            # batch mode
            assert t.numel() == length.sum(), "texts with length: {} does not match declared length: {}".format(t.numel(), length.sum())
            texts = []
            index = 0
            for i in range(length.numel()):
                l = length[i]
                texts.append(
                    self.decode(
                        t[index:index + l ], torch.LongTensor([l])))
                index += l
            return texts

class averager(object):
    """Compute average for `torch.Variable` and `torch.Tensor`. """

    def __init__(self):
        self.reset()

    def add(self, v):
        if isinstance(v, Variable):
            count = v.data.numel()
            v = v.data.sum()
        elif isinstance(v, torch.Tensor):
            count = v.numel()
            v = v.sum()

        self.n_count += count
        self.sum += v

    def reset(self):
        self.n_count = 0
        self.sum = 0

    def val(self):
        res = 0
        if self.n_count != 0:
            res = self.sum / float(self.n_count)
        return res

def loadData(v, data):
    #v.data.resize_(data.size()).copy_(data)
    v.resize_(data.size()).copy_(data)  # for pytorch 1.1.0
