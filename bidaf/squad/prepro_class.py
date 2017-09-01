import json
import os
# data: q, cq, (dq), (pq), y, *x, *cx
# shared: x, cx, (dx), (px), word_counter, char_counter, word2vec
# no metadata
from collections import Counter

from tqdm import tqdm
import nltk.tokenize as nltk
from squad.utils import get_word_span, get_word_idx, process_tokens

import nltk.tokenize as nltk
import os

class PreproClass():
    
    def __init__(self):
        
        self.args = self.get_args()
        self.w2v_emb = self.getW2VEmb()

    def getW2VEmb(self):
        q2qw2v_path = self.args['q2qw2v_path']
        total = len(q2qw2v_path)
        word2vec_dict = {}
        with open(q2qw2v_path, 'r', encoding='utf-8') as fh:
            for line in tqdm(fh, total=total):
                array = line.lstrip().rstrip().split(" ")
                word = array[0]
                vector = list(map(float, array[1:]))
                word2vec_dict[word] = vector
        return word2vec_dict
    
    def get_args(self):
        args = {}
        # q2qw2v_path
        args['q2qw2v_path'] = 'data/q2q.w2v.300d.txt'
        return args
    
    # 基于qp model retrain 的emb
    def get_word2vec_q2qw2v(self, word_counter):
        q2qw2v_path = self.args['q2qw2v_path']
        total = len(q2qw2v_path)
        word2vec_dict = {}
        for word in word_counter:
            if word in self.w2v_emb:
                word2vec_dict[word] = self.w2v_emb[word]
            elif word.capitalize() in self.w2v_emb:
                word2vec_dict[word.capitalize()] = self.w2v_emb[word.capitalize()]
            elif word.lower() in self.w2v_emb:
                word2vec_dict[word.lower()] = self.w2v_emb[word.lower()]
            elif word.upper() in self.w2v_emb:
                word2vec_dict[word.upper()] = self.w2v_emb[word.upper()]

        print("{}/{} of word vocab have corresponding vectors in {}".format(len(word2vec_dict), len(word_counter), q2qw2v_path))
        return word2vec_dict

    # data_file online.json
    def prepro_online(self, source_data, start_ratio=0.0, stop_ratio=1.0, in_path=None):
            
        #sent_tokenize = nltk.sent_tokenize
        def word_tokenize(tokens):
            return [token.replace("''", '"').replace("``", '"') for token in nltk.word_tokenize(tokens)]

        q, cq, y, rx, rcx, ids, idxs = [], [], [], [], [], [], []
        na = []
        cy = []
        x, cx = [], []
        p = []
        word_counter, char_counter, lower_word_counter = Counter(), Counter(), Counter()
        start_ai = int(round(len(source_data['data']) * start_ratio))
        stop_ai = int(round(len(source_data['data']) * stop_ratio))
        for ai, article in enumerate(tqdm(source_data['data'][start_ai:stop_ai])):
            xp, cxp = [], []
            pp = []
            x.append(xp)    # [[[xi], []], []]所有article   [[[xi], [xi]]]
            cx.append(cxp)
            p.append(pp)
            for pi, para in enumerate(article['paragraphs']):   # 每个段落
                # wordss
                context = para['context']
                context = context.replace("''", '" ')
                context = context.replace("``", '" ')
                xi = list(map(word_tokenize, [context]))       # paragraph context
                xi = [process_tokens(tokens) for tokens in xi]  # process tokens

                # given xi, add chars
                cxi = [[list(xijk) for xijk in xij] for xij in xi]
                xp.append(xi)           # paragraph list (a article)
                cxp.append(cxi)
                pp.append(context)

                for xij in xi:
                    for xijk in xij:
                        word_counter[xijk] += len(para['qas'])
                        lower_word_counter[xijk.lower()] += len(para['qas'])
                        for xijkl in xijk:
                            char_counter[xijkl] += len(para['qas']) # context 中的每个token，权重为question的个数

                rxi = [ai, pi]
                assert len(x) - 1 == ai
                assert len(x[ai]) - 1 == pi
                for qa in para['qas']:
                    # get words
                    qi = word_tokenize(qa['question'])
                    qi = process_tokens(qi)
                    cqi = [list(qij) for qij in qi]

                    for qij in qi:
                        word_counter[qij] += 1
                        lower_word_counter[qij.lower()] += 1
                        for qijk in qij:
                            char_counter[qijk] += 1

                    q.append(qi)
                    cq.append(cqi)
                    rx.append(rxi)
                    rcx.append(rxi)
                    ids.append(qa['id'])
                    idxs.append(len(idxs))

        #word2vec_dict = get_word2vec(args, word_counter)
        #lower_word2vec_dict = get_word2vec(args, lower_word_counter)

        # get_word2vec_q2pw2v_emb EQnA data emb
        #word2vec_dict = get_word2vec_q2pw2v_emb(args, word_counter)
        #lower_word2vec_dict = get_word2vec_q2pw2v_emb(args, lower_word_counter)

        # get_word2vec_q2qw2v EQnA model emb
        word2vec_dict = self.get_word2vec_q2qw2v(word_counter)
        lower_word2vec_dict = self.get_word2vec_q2qw2v(lower_word_counter)

        # add context here
        data = {'q': q, 'cq': cq, '*x': rx, '*cx': rcx,
                'idxs': idxs, 'ids': ids, '*p': rx,}
        
        print ('id list len = ', len(ids))
        # q:    question token list
        # cq:   question token charchater list
        # rx:   [article_id, paragraph_id]
        # rcx:  [article_id, paragraph_id]
        # ids:  question id list
        # idxs: question id list(start from 0)

        shared = {'x': x, 'cx': cx, 'p': p,
                  'word_counter': word_counter, 'char_counter': char_counter, 'lower_word_counter': lower_word_counter,
                  'word2vec': word2vec_dict, 'lower_word2vec': lower_word2vec_dict}
        
        # x:            context tokens list  [  art[  cont[   seq[] ]            ]]
        # cx:           context tokens character list
        # p:            context [["xxx, "xxx"], []]
        # word_counter: context+question word_count
        # lower_word_counter: 
        # char_counter: context+question word_ch_count
        # word2vec:
        # lower_word2vec:

        return data, shared

if __name__ == "__main__":
    main()