#!/usr/bin/env python
# coding: utf-8

# In[35]:


from collections import Counter
from pycocotools.coco import COCO
import nltk
nltk.download('punkt')
import pickle
import argparse


# In[36]:


class Vocabulary:
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.index = 0
        
    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.index
            self.idx2word[self.index] = word
            self.index += 1
            
    def __call__(self,word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]
    
    def __len__(self):
        return len(self.word2idx)


# In[37]:


def build_vocab(json,threshold):
    coco = COCO(json)
    counter = Counter()
    ids = coco.anns.keys()
    
    for i,no in enumerate(ids):
        caption = str(coco.anns[no]['caption'])
        tokens = nltk.tokenize.word_tokenize(caption.lower())
        counter.update(tokens)
        
        if (i+1) % 1000 == 0:
            print("[{}/{}] Tokenized the captions ".format(i+1,len(ids)))
    
    words = [word for word,cnt in counter.items() if cnt>=threshold]
    
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')
                  
    for i,word in enumerate(words):
        vocab.add_word(word)
        
    return vocab


# In[38]:


def main(args):
    vocab = build_vocab(json = args.json_path , threshold = args.threshold)
    vocab_path = args.vocab_path
    with open(vocab_path,'wb') as f:
        pickle.dump(vocab , f)
    
    print('total vocabulary size is {}'.format(len(vocab)))
    print('saved the vocabulary to {}'.format(vocab_path))
    


# In[39]:


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_path',type=str,
                       default='data/annotations/captions_train2014.json',
                       help = 'path for train in annotation file')
    parser.add_argument('--vocab_path',type = str,
                       default = 'data/vocab.pkl',
                       help = 'path for vocabulary wrapper')
    parser.add_argument('--threshold',type = int ,
                       default = 4,
                       help ='minimum word count threshold')
    args = parser.parse_args(args=[])
    main(args)

