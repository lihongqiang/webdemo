import os

import tensorflow as tf

import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

flags = tf.app.flags

# Names and directories
# flags.DEFINE_string("model_name", "squad", "Model name [basic]")
flags.DEFINE_string("model_name", "EQnA", "Model name [basic]")

# flags.DEFINE_string("data_dir", "data/squad", "Data dir [data/squad]")# 数据路径
#flags.DEFINE_string("data_dir", "data/EQnA", "Data dir [data/squad]")# 数据路径

#flags.DEFINE_string("run_id", "0", "Run ID [0]")
flags.DEFINE_string("forward_name", "single", "Forward name [single]")
flags.DEFINE_string("out_base_dir", "out", "out base dir [out]")
flags.DEFINE_string("out_dir", "", "out dir [out/date]")
flags.DEFINE_string("answer_path", "", "Answer path []")# 设置答案的路径
flags.DEFINE_string("eval_path", "", "Eval path []")    # 设置评价脚本的路径
flags.DEFINE_string("shared_path", "", "Shared path []")# 生成的共享文件路径

# test
flags.DEFINE_string("load_path", "", "Load path []")    # 设置加载的模型路径
flags.DEFINE_integer("load_step", 0, "load step [0]")   # step

# Device placement
flags.DEFINE_string("device", "/cpu:0", "default device for summing gradients. [/cpu:0]")
flags.DEFINE_string("device_type", "gpu", "device for computing gradients (parallelization). cpu | gpu [gpu]")
flags.DEFINE_integer("num_gpus", 1, "num of gpus or cpus for computing gradients [1]")

# Essential training and test options
flags.DEFINE_string("mode", "test", "trains | test | forward [test]")
flags.DEFINE_boolean("load", True, "load saved data? [True]")
flags.DEFINE_bool("single", False, "supervise only the answer sentence? [False]")
flags.DEFINE_boolean("debug", False, "Debugging mode? [False]")
flags.DEFINE_bool('load_ema', True, "load exponential average of variables when testing?  [True]")
flags.DEFINE_bool("eval", True, "eval? [True]")
flags.DEFINE_bool("wy", False, "Use wy for loss / eval? [False]")
flags.DEFINE_bool("na", False, "Enable no answer strategy and learn bias? [False]")
flags.DEFINE_float("th", 0.5, "Threshold [0.5]")

# top K answer
flags.DEFINE_integer("topk", 0, "score top k")

# online
flags.DEFINE_boolean("online", False, "online")

# sentece token
flags.DEFINE_boolean("sentece_token", False, "sentece")

# Training / test parameters
flags.DEFINE_integer("batch_size", 60, "Batch size [60]")
flags.DEFINE_integer("val_num_batches", 100, "validation num batches [100]")
flags.DEFINE_integer("test_num_batches", 0, "test num batches [0]")
flags.DEFINE_integer("num_epochs", 12, "Total number of epochs for training [12]")
flags.DEFINE_integer("num_steps", 30000, "Number of steps [20000]")

flags.DEFINE_float("init_lr", 0.001, "Initial learning rate [0.001]")
flags.DEFINE_float("input_keep_prob", 0.8, "Input keep prob for the dropout of LSTM weights [0.8]")
flags.DEFINE_float("keep_prob", 0.8, "Keep prob for the dropout of Char-CNN weights [0.8]")
flags.DEFINE_float("wd", 0.0, "L2 weight decay for regularization [0.0]")
flags.DEFINE_integer("hidden_size", 100, "Hidden size [100]")
flags.DEFINE_integer("char_out_size", 100, "char-level word embedding size [100]")
flags.DEFINE_integer("char_emb_size", 8, "Char emb size [8]")
flags.DEFINE_string("out_channel_dims", "100", "Out channel dims of Char-CNN, separated by commas [100]")
flags.DEFINE_string("filter_heights", "5", "Filter heights of Char-CNN, separated by commas [5]")
flags.DEFINE_bool("finetune", False, "Finetune word embeddings? [False]")

flags.DEFINE_integer("highway_num_layers", 2, "highway num layers [2]")
flags.DEFINE_bool("share_cnn_weights", True, "Share Char-CNN weights [True]")
flags.DEFINE_bool("share_lstm_weights", True, "Share pre-processing (phrase-level) LSTM weights [True]")
flags.DEFINE_float("var_decay", 0.999, "Exponential moving average decay for variables [0.999]")

# Optimizations
flags.DEFINE_bool("cluster", False, "Cluster data for faster training [False]")
flags.DEFINE_bool("len_opt", False, "Length optimization? [False]")
flags.DEFINE_bool("cpu_opt", False, "CPU optimization? GPU computation can be slower [False]")

# Logging and saving options
flags.DEFINE_boolean("progress", True, "Show progress? [True]")
flags.DEFINE_integer("log_period", 100, "Log period [100]")
flags.DEFINE_integer("eval_period", 1000, "Eval period [1000]")
flags.DEFINE_integer("save_period", 1000, "Save Period [1000]")
flags.DEFINE_integer("max_to_keep", 20, "Max recent saves to keep [20]")
flags.DEFINE_bool("dump_eval", True, "dump eval? [True]")
flags.DEFINE_bool("dump_answer", True, "dump answer? [True]")
flags.DEFINE_bool("vis", False, "output visualization numbers? [False]")
flags.DEFINE_bool("dump_pickle", True, "Dump pickle instead of json? [True]")
flags.DEFINE_float("decay", 0.9, "Exponential moving average decay for logging values [0.9]")

# Thresholds for speed and less memory usage
flags.DEFINE_integer("word_count_th", 10, "word count th [100]")
flags.DEFINE_integer("char_count_th", 50, "char count th [500]")
flags.DEFINE_integer("sent_size_th", 400, "sent size th [64]")
flags.DEFINE_integer("num_sents_th", 8, "num sents th [8]")
flags.DEFINE_integer("ques_size_th", 30, "ques size th [32]")
flags.DEFINE_integer("word_size_th", 16, "word size th [16]")
flags.DEFINE_integer("para_size_th", 256, "para size th [256]")

# Advanced training options
flags.DEFINE_bool("lower_word", True, "lower word [True]")
flags.DEFINE_bool("squash", False, "squash the sentences into one? [False]")
flags.DEFINE_bool("swap_memory", True, "swap memory? [True]")
flags.DEFINE_string("data_filter", "max", "max | valid | semi [max]")
flags.DEFINE_bool("use_glove_for_unk", True, "use glove for unk [False]")
flags.DEFINE_bool("known_if_glove", True, "consider as known if present in glove [False]")
flags.DEFINE_string("logit_func", "tri_linear", "logit func [tri_linear]")
flags.DEFINE_string("answer_func", "linear", "answer logit func [linear]")
flags.DEFINE_string("sh_logit_func", "tri_linear", "sh logit func [tri_linear]")


# Ablation options 
flags.DEFINE_bool("q2c_att", True, "question-to-context attention? [True]")
flags.DEFINE_bool("c2q_att", True, "context-to-question attention? [True]")
flags.DEFINE_bool("dynamic_att", False, "Dynamic attention [False]")

# Ablation options 
flags.DEFINE_bool("use_char_emb", True, "use char emb? [True]")
flags.DEFINE_bool("use_word_emb", True, "use word embedding? [True]")
flags.DEFINE_bool("highway", True, "Use highway? [True]")

# addition 
flags.DEFINE_bool("use_sentence_emb", False, "use sentence emb? [True]")
flags.DEFINE_integer("sent_dim", 600, "sentence emb dim")

# test
flags.DEFINE_string("data_dir", 'data/input/', "input file path")
flags.DEFINE_string("model_dir", 'model/01-08-2017/', "trained model path")
flags.DEFINE_string("output_dir", 'data/output/', "output file path")
flags.DEFINE_integer("num", 1, "answer number")
flags.DEFINE_string("input_suffix", "tsv", "Filename suffix of data.")
flags.DEFINE_integer("id_index", -1, "the index of Id")
flags.DEFINE_integer("query_index", -1, "answer number")
flags.DEFINE_integer("context_index", -1, "answer number")

# trans to data json
import pandas as pd
import hashlib
import json
import os
import time
import requests
import threading
import sys
from squad.prepro_class import PreproClass
from basic.main import main as m
ISOTIMEFORMAT='%Y-%m-%d %X'

config = flags.FLAGS

class ServeClass():
    
    def __init__(self):
        self.prepro = PreproClass()

    # input:
    #   file_path: tsv file including (Query Context phrase)
    #   output_dir: file dir
    # output:
    #   file_path: generated file path
    def generateJson(self, file_path):
        
        def GetHashCode(context):
            hash = hashlib.md5()
            hash.update(context.encode('utf-8'))
            return hash.hexdigest()
        
        online_data = pd.read_csv(file_path, header=None, sep='\t', dtype=str).fillna('')
        if config.id_index != -1:
            online_data.rename_axis({config.id_index: 'Id', config.query_index:'Query', config.context_index:'Context'}, axis=1, inplace=True)
        else:
            online_data.rename_axis({config.query_index:'Query', config.context_index:'Context'}, axis=1, inplace=True)
            online_data['Id'] = online_data.apply(lambda row: GetHashCode(row['Context'] + ' ' + row['Query']), axis=1)
        #online_data = pd.DataFrame({"Query":query, "Context":context, "phrase":answer}, columns=["Query", "Context", "phrase"], index=[0])
        # 只处理lable为1的row
        target_dev_data = {}
        target_dev_data['version'] = '1.1'
        target_dev_data['data'] = list()

        item = {}
        item['paragraphs'] = list()
        item['title'] = 'Online'

        def transEachRow(row, paragraphs):
            # 新生成的context
            paragraph = {}
            paragraph['qas'] = list()
            paragraph['context'] = row['Context'].strip()

            phrase = {}
            #phrase['id'] = GetHashCode(row['Context'] + ' ' + row['Query'])
            phrase['id'] = row['Id']
            phrase['question'] = row['Query'].strip()

            paragraph['qas'].append(phrase)

            paragraphs.append(paragraph)

        online_data.apply(transEachRow, axis=1, args=[item['paragraphs']])

        target_dev_data['data'].append(item)
        return target_dev_data, online_data
    
    

    def testData(self, num, data, shared, output_dir, model_dir):
        return m(config, num, data, shared, output_dir, model_dir)
    
    # save answer
    def saveAnswer(self, online_data, id2answer_dict, output_dir):
     
        online_data['Answer'] = online_data.apply(lambda row: '|||'.join([ ans+":::"+score for ans, score in zip(id2answer_dict[row['Id']].split('|||'), id2answer_dict['scores'][row['Id']].split('|||'))]), axis=1)
        
        online_data.to_csv(os.path.join(output_dir, 'output.tsv'), sep='\t', header=False, index=False)
      
        
    def getAnswerPhrase(self, file_path, output_dir, model_dir, num):
        
        # <1s
        print ('build json', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        data, online_data = self.generateJson(file_path)

        # <1s search twice glove for word embedding
        print ('build data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        data, shared = self.prepro.prepro_online(data)

        # 3s run the model in GPU
        print ('test data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        id2answer_dict = self.testData(num, data, shared, output_dir, model_dir)

        # <1s
        print ('save data', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        
        self.saveAnswer(online_data, id2answer_dict, output_dir)
        print ('finish ', time.strftime( ISOTIMEFORMAT, time.localtime( time.time() )))
        
        #self.AnswerByBiDAF = ans
        #return ans
        
def get_input_data(data_dir, suffix):
    file_list = os.listdir(data_dir)
    data_path = ""
    for fn in file_list:
        if fn.endswith(suffix):
            data_path = os.path.join(data_dir, fn)
            return data_path
    return data_path

def main(_):
    
    assert config.num >= 0, ("--num must >= 0")
    assert config.input_suffix == 'tsv', ("--input_suffix must be tsv")
    # assert config.id_index >= 0, ("--id_index must >= 0")
    assert config.query_index >= 0, ("--query_index must >= 0")
    assert config.context_index >= 0, ("--context_index must >= 0")
    
    serve = ServeClass()
    file_path = get_input_data(config.data_dir, config.input_suffix)
    serve.getAnswerPhrase(file_path, config.output_dir, config.model_dir, config.num)
    
if __name__ == "__main__":
    tf.app.run()
