import argparse
import json
import math
import os
import shutil
from pprint import pprint

import tensorflow as tf
from tqdm import tqdm
import numpy as np

from basic.evaluator import ForwardEvaluator, MultiGPUF1Evaluator
from basic.graph_handler import GraphHandler
from basic.model import get_multi_gpu_models
from basic.trainer import MultiGPUTrainer
from basic.read_data import read_data, get_squad_data_filter, update_config, update_config_online
from my.tensorflow import get_num_params

class BiDAF():
    def __init__(self, config):
        
        config.len_opt = True
        config.cluster = True
        config.dump_eval = False
        config.online = True
        
        config.out_dir = config.model_dir 
        
        config.answer_path = os.path.join(config.output_dir, 'output.json')
        config.eval_path = os.path.join(config.output_dir, 'eval.json')
        self.set_dirs(config)
        
        update_config_online(config)
        self.models = get_multi_gpu_models(config)
        self.model = self.models[0]
        self.evaluator = MultiGPUF1Evaluator(config, self.models, tensor_dict=self.models[0].tensor_dict if config.vis else None)
        self.graph_handler = GraphHandler(config, self.model)
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options = tf.GPUOptions(allow_growth = True)))
        self.graph_handler.initialize(self.sess)
        
    def main(self, config, num, data, shared, output_dir, model_dir):
        config.topk = num
        config.data = data
        config.shared = shared
        return self._test(config)

    def set_dirs(self, config):
        # create directories
        assert config.load or config.mode == 'train', "config.load must be True if not training"
        if not config.load and os.path.exists(config.out_dir):
            shutil.rmtree(config.out_dir)

        config.save_dir = os.path.join(config.out_dir, "save")
        print ('create save_dir', config.save_dir)

        config.log_dir = os.path.join(config.out_dir, "log")
        config.eval_dir = os.path.join(config.out_dir, "eval")
        config.answer_dir = os.path.join(config.out_dir, "answer")
        if not os.path.exists(config.out_dir):
            os.makedirs(config.out_dir)
        if not os.path.exists(config.save_dir):
            os.mkdir(config.save_dir)
        if not os.path.exists(config.log_dir):
            os.mkdir(config.log_dir)
        if not os.path.exists(config.answer_dir):
            os.mkdir(config.answer_dir)
        if not os.path.exists(config.eval_dir):
            os.mkdir(config.eval_dir)


    def _config_debug(self, config):
        if config.debug:
            config.num_steps = 2
            config.eval_period = 1
            config.log_period = 1
            config.save_period = 1
            config.val_num_batches = 2
            config.test_num_batches = 2

    def _test(self, config):
        test_data = read_data(config, True)
        
        #update_config(config, [test_data])

        self._config_debug(config)

        if config.use_glove_for_unk:
            word2vec_dict = test_data.shared['lower_word2vec'] if config.lower_word else test_data.shared['word2vec']
            new_word2idx_dict = test_data.shared['new_word2idx']
            idx2vec_dict = {idx: word2vec_dict[word] for word, idx in new_word2idx_dict.items()}
            new_emb_mat = np.array([idx2vec_dict[idx] for idx in range(len(idx2vec_dict))], dtype='float32')
            config.new_emb_mat = new_emb_mat

        # pprint(config.__flags, indent=2)


        num_steps = math.ceil(1.0 * test_data.num_examples / (config.batch_size * config.num_gpus)) # 2021 / 10 = 203

        # 这个地方可以自己设置test的num batch，就是不测试所有的batch，一般小于总大小
        if 0 < config.test_num_batches < num_steps:
            num_steps = config.test_num_batches


        e = None
        for multi_batch in tqdm(test_data.get_multi_batches(config.batch_size, config.num_gpus, num_steps=num_steps, cluster=config.cluster), total=num_steps):
            ei = self.evaluator.get_evaluation(self.sess, multi_batch)
            e = ei if e is None else e + ei
            if config.vis:
                eval_subdir = os.path.join(config.eval_dir, "{}-{}".format(ei.data_type, str(ei.global_step).zfill(6)))
                if not os.path.exists(eval_subdir):
                    os.mkdir(eval_subdir)
                path = os.path.join(eval_subdir, str(ei.idxs[0]).zfill(8))
                self.graph_handler.dump_eval(ei, path=path)

        return e.id2answer_dict


