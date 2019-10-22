#start_of_genes_definitions
#key=data;  type=random_array_of_fields;  length=13
#key=fields_to_use;  type=random_int;  from=13;  to=13;  step=1
#key=map_dict;  type=random_from_set;  set=True
#key=field_ev_prefix;  type=random_from_set;  set=ev_field_trnlp
#key=field_ev_prefix_use_source_names;  type=random_from_set;  set=True
#key=n_gpu;  type=random_int;  from=0;  to=0;  step=1
#key=nfolds;  type=random_int;  from=3;  to=3;  step=1
#key=random_folds;  type=random_from_set;  set=True
#key=random_folds_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=use_validation_set;  type=random_from_set;  set=True
#key=random_valid;  type=random_from_set;  set=True
#key=random_valid_size;  type=random_float;  from=0.3;  to=0.3;  step=0.1
#key=random_valid_folds;  type=random_int;  from=10;  to=10;  step=1
#key=random_seed_init;  type=random_int;  from=1;  to=10000000;  step=1
#key=filter_column;  type=random_from_set;  set=field|field.csv
#key=train_set_from;  type=random_from_set;  set=self.timestamp('2013-11-01')
#key=train_set_to;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_from;  type=random_from_set;  set=self.timestamp('2014-11-01')
#key=valid_set_to;  type=random_from_set;  set=self.timestamp('2016-11-01')
#key=filter_column_2;  type=random_from_set;  set=
#key=train_set_from_2;  type=random_from_set;  set=
#key=train_set_to_2;  type=random_from_set;  set=
#key=valid_set_from_2;  type=random_from_set;  set=
#key=valid_set_to_2;  type=random_from_set;  set=
#key=include_columns_type;  type=random_from_set;  set=is_dict_only
#key=include_columns_containing;  type=random_from_set;  set=
#key=ignore_columns_containing;  type=random_from_set;  set=%ev_field%
#key=binary_balancing;  type=random_from_set;  set=False
#key=binary_balancing_0;  type=random_float;  from=0.1;  to=1;  step=0.02
#key=binary_balancing_1;  type=random_float;  from=1;  to=1;  step=0.02
#key=binary_eval_fun;  type=random_from_set;  set='ROCAUC','PRCAUC'
#key=transformer_model;  type=random_from_set;  set='bert','xlnet','xlm','roberta','distilbert'
#key=transformer_model_id;  type=random_int;  from=0;  to=9;  step=1
#key=objective_multiclass;  type=random_from_set;  set='multiclass'
#key=objective_regression;  type=random_from_set;  set='regression_l1'
#key=epoch;  type=random_int;  from=1;  to=3;  step=1
#key=max_train_steps;  type=random_int;  from=0;  to=0;  step=1
#key=learning_rate;  type=random_float;  from=0.00004;  to=0.1;  step=0.00001
#key=gradient_accumulation_steps;  type=random_int;  from=1;  to=10;  step=1
#key=train_batch_size;  type=random_int;  from=5;  to=20;  step=1
#key=max_seq_length;  type=random_int;  from=128;  to=128;  step=1
#key=weight_decay;  type=random_float;  from=0;  to=1;  step=0.1
#key=adam_epsilon;  type=random_float;  from=1e-8;  to=1e-8;  step=0.1
#key=warmup_ratio;  type=random_float;  from=0;  to=1;  step=0.01
#key=warmup_steps;  type=random_int;  from=0;  to=10;  step=1
#key=max_grad_norm;  type=random_float;  from=0;  to=2;  step=0.01
#key=fp16; type=random_from_set;  set=False
#key=fp16_opt_level;  type=random_from_set;  set='O0','O1','O2','O3'
#key=logging_steps;  type=random_int;  from=10;  to=10;  step=1
#key=tokenizer_do_lower_case; type=random_from_set;  set=True,False
#key=clean_text_v;  type=random_int;  from=0;  to=3;  step=1
#key=start_fold;  type=random_from_set;  set=0
#key=num_threads;  type=random_int;  from=4;  to=4;  step=1
#key=use_float32_dtype; type=random_from_set;  set=True
#key=min_perf_criteria;  type=random_float;  from=0.55;  to=0.55;  step=0.1
#key=print_to_html; type=random_from_set;  set=True
#key=print_tables; type=random_from_set;  set=False
#end_of_genes_definitions

# AICHOO OS Evolving Agent 
# Documentation about AIOS and how to create Evolving Agents can be found on our WiKi
# https://github.com/eghamtech/AIOS/wiki/Evolving-Agents
# https://github.com/eghamtech/AIOS/wiki/AI-OS-Introduction

# this agent concatenates given columns into sentences by rows 
# and applies one of the Transformers NLP models to learn the target
#
# if column is a text one and map_dict==True it loads corresponding text from its dictionary
# if column is just numeric or map_dict==False then the number is converted into text

# Adapted from HuggingFace code
# 
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy  as np
import math, os.path, bz2, pickle, logging
import dateutil, calendar
import random as rn
import json, re

from sklearn.metrics import roc_auc_score, precision_score, accuracy_score, log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error

from sklearn.utils.extmath import softmax
from sklearn.model_selection import StratifiedShuffleSplit

from math import sqrt
from datetime import datetime

from tqdm import tqdm, trange
from __future__ import absolute_import, division, print_function

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler, TensorDataset)
from torch.utils.data.distributed import DistributedSampler

from transformers import (  WEIGHTS_NAME, 
                            BertConfig,       BertForSequenceClassification,       BertTokenizer,
                            RobertaConfig,    RobertaForSequenceClassification,    RobertaTokenizer,
                            XLMConfig,        XLMForSequenceClassification,        XLMTokenizer, 
                            XLNetConfig,      XLNetForSequenceClassification,      XLNetTokenizer,
                            DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)

from transformers import AdamW, WarmupLinearSchedule

ALL_MODELS = sum((tuple(conf.pretrained_config_archive_map.keys()) for conf in (BertConfig, XLNetConfig, XLMConfig, 
                                                                                RobertaConfig, DistilBertConfig)), ())

MODEL_CLASSES = {
    'bert':       (BertConfig,       BertForSequenceClassification,       BertTokenizer),
    'xlnet':      (XLNetConfig,      XLNetForSequenceClassification,      XLNetTokenizer),
    'xlm':        (XLMConfig,        XLMForSequenceClassification,        XLMTokenizer),
    'roberta':    (RobertaConfig,    RobertaForSequenceClassification,    RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForSequenceClassification, DistilBertTokenizer)
}

MODEL_NAMES = {
    'bert'      : ['bert-base-uncased', 'bert-large-uncased', 'bert-base-cased', 
                   'bert-large-cased',  'bert-base-multilingual-cased', 'bert-large-uncased-whole-word-masking',
                   'bert-large-cased-whole-word-masking','bert-large-uncased-whole-word-masking-finetuned-squad',
                   'bert-large-cased-whole-word-masking-finetuned-squad','bert-base-cased-finetuned-mrpc'],
    'xlnet'     : ['xlnet-base-cased','xlnet-large-cased'],
    'xlm'       : ['xlm-mlm-en-2048','xlm-mlm-ende-1024','xlm-mlm-enfr-1024',
                   'xlm-mlm-enro-1024','xlm-mlm-xnli15-1024',
                   'xlm-mlm-tlm-xnli15-1024','xlm-clm-enfr-1024','xlm-clm-ende-1024',
                   'xlm-mlm-17-1280','xlm-mlm-100-1280'],
    'roberta'   : ['roberta-base','roberta-large','roberta-large-mnli'],
    'distilbert': ['distilbert-base-uncased','distilbert-base-uncased-distilled-squad','distilgpt2']
}

class cls_ev_agent_{id}:
    logger    = logging.getLogger(__name__)
    logging.basicConfig(format  = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt = '%Y/%m/%d %H:%M:%S',
                        level = logging.INFO)

    # obtain a unique ID for the current instance
    result_id   = {id}
    # obtain random field (same for all instances within the evolution) which will be the prediction target for this instance/evolution
    target_definition = "{field_to_predict}"
    # field definition received from the kernel contains two parts: name of the field and CSV filename that holds the actual data
    # load these two parts into variables
    target_col  = target_definition.split("|")[0]
    target_file = target_definition.split("|")[1]

    # obtain random selection of fields; number of fields to be selected specified in data:length gene for this instance
    data_defs     = {data}
    fields_to_use = {fields_to_use}
    start_fold    = {start_fold}
    nfolds        = {nfolds}

    map_dict      = {map_dict}
    clean_text_v  = {clean_text_v}
    field_ev_prefix_use_source_names = {field_ev_prefix_use_source_names}
    
    num_threads   = {nthread}
    rn_seed_init  = {random_seed_init}
    n_gpu         = {n_gpu}

    params        = {}         # all parameters
    params['algo']= {}         # ML algo parameters
    dicts_agent   = {}         # various dictionary to be saved as part of model
    
    # if filter columns are specified then training and validation sets will be selected based on filter criteria
    # based on filter criteria training + validation sets will not necessarily constitute all data, the remainder will be called "test set"
    filter_column   = "{filter_column}"
    filter_column_2 = "{filter_column_2}"
    
    # fields matching the specified prefix will not be used in the model
    ignore_columns_containing  = "{ignore_columns_containing}"
    # include only fields matching string e.g., only properly scaled columns should be used with MLP
    include_columns_containing = "{include_columns_containing}"
    
    objective_multiclass = {objective_multiclass}
    objective_regression = {objective_regression}
    
    print_tables  = {print_tables}
    print_to_html = {print_to_html}

    use_validation_set = {use_validation_set}
    use_float32_dtype  = {use_float32_dtype}
    min_perf_criteria  = {min_perf_criteria}

    def set_seed(self, seed_init):
        rn.seed(seed_init)
        np.random.seed(seed_init)
        torch.manual_seed(seed_init)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed_init)
   
    def __init__(self):        
        self.set_seed(self.rn_seed_init)        # set same seed for every run of this agent's instance
        
        # remove the target field for this instance from the data used for training
        if self.target_definition in self.data_defs:
            self.data_defs.remove(self.target_definition)
        
        # create new field name based on "field_ev_prefix" with unique instance ID
        # and filename to save new field data    
        self.field_ev_prefix = "{field_ev_prefix}"
        if self.field_ev_prefix_use_source_names:                   
            # concatenate all source column names into new field prefix
            col_max_length = int(200 / self.fields_to_use)             # allow 200 characters max combined col name length
            for i in range(0,self.fields_to_use):
                col_name = self.data_defs[i].split("|")[0]
                col_name = col_name[:col_max_length]                   # only take first col_max_length chars from each column
                self.field_ev_prefix = self.field_ev_prefix + '_' + col_name
        
        self.output_column   = self.field_ev_prefix + '_' + str(self.result_id)
        self.output_filename = self.output_column + ".csv"
        
        self.model_env_init()

        s_file = workdir + self.output_column + '_dicts.model'
        if os.path.isfile(s_file):
            rfile = bz2.BZ2File(s_file, 'r')
            self.dicts_agent = pickle.load(rfile)
            rfile.close()
            print (str(datetime.now()), self.output_column + ' dictionaries model loaded')

            # if saved model for the target field already exists then load it from filesystem
            self.predictors = []
            if self.dicts_agent['params']['random_folds'] == False:
                from_fold = self.start_fold
                to_fold   = self.nfolds
            else:
                from_fold = 0
                to_fold   = 3  # use fixed 3 saved models to make any prediction

            for fold in range(from_fold, to_fold):
                s_file = workdir + self.output_column + "_fold" + str(fold) + ".model"
                if os.path.exists(s_file):
                    predictor_stored = self.model_load(s_file)
                    self.predictors.append(predictor_stored)
                    print (str(datetime.now()), self.output_column + ' fold ' + str(fold) + ' predictor model loaded')  
                      
        # obtain columns definitions to filter data set by
        if self.is_set(self.filter_column):
            self.filter_filename = self.filter_column.split("|")[1]
            self.filter_column   = self.filter_column.split("|")[0]
      
        if self.is_set(self.filter_column_2):
            self.filter_filename_2 = self.filter_column_2.split("|")[1]
            self.filter_column_2   = self.filter_column_2.split("|")[0]

    
    def is_set(self, s):
        return len(s)>0 and s!="0"

    def is_use_column(self, s):
        # AIOS Kernel now selects columns using agent parameters
        # so no need to filter inside the agent       
        if s.find(self.target_col)>=0:  # ignore columns that contain target_col as they are a derivative of the target
            return False 
        
        return True

    def timestamp(self, x):
        return calendar.timegm(dateutil.parser.parse(x).timetuple())

    def print_tbl(self, mesg):
        if self.print_tables:
            print (mesg)
    
    def print_html(self, df, max_rows=50, max_cols=30, jup_notebook=True):
        if self.print_to_html:
            print (df.to_html(max_rows=max_rows,max_cols=max_cols))
        elif jup_notebook:
            display (df)
        else:
            print (df)
    
    def list_mean(self, lst, precision=4):
        return np.round(sum(lst)/float(len(lst)), decimals=precision)
    
    def prc_auc(self, train_y, pred):  
        precision, recall, thresholds = precision_recall_curve(train_y, pred)  
        prc_auc = auc(recall, precision)
                     
        return prc_auc 

    def map_from_continuous_set_to_set(self, s1, s2, s1_value):
        # maps specified value s1_value 
        # from one continuous set of integers 0 to s1
        # to smaller continuous set of integers from 0 to s2
        s1_chunks = np.array_split(range(s1), s2)

        for i in range(s2):
            if s1_value in s1_chunks[i]:
                return i
        return 0  # return 0 if something wrong and s1_value not found in split set 0:s1
    
    def clean_text_v1(self, s):
        s = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", s)     
        s = re.sub(r"\'s", " 's", s) 
        s = re.sub(r"\'ve", " \'ve", s) 
        s = re.sub(r"n\'t", " n\'t", s) 
        s = re.sub(r"\'re", " \'re", s) 
        s = re.sub(r"\'d", " \'d", s) 
        s = re.sub(r"\'ll", " \'ll", s) 
        s = re.sub(r",", " , ", s) 
        s = re.sub(r"!", " ! ", s) 
        s = re.sub(r"\(", " ( ", s) 
        s = re.sub(r"\)", " ) ", s) 
        s = re.sub(r"\?", " ? ", s) 
        s = re.sub(r"\s{2,}", " ", s)       
        s = s.strip().lower()
        return  s
    
    def clean_text_v2(self, s):
        # Replace numbers and symbols with language
        s = s.replace('&', ' and ')
        s = s.replace('@', ' at ')
        s = s.replace('0', ' zero ')
        s = s.replace('1', ' one ')
        s = s.replace('2', ' two ')
        s = s.replace('3', ' three ')
        s = s.replace('4', ' four ')
        s = s.replace('5', ' five ')
        s = s.replace('6', ' six ')
        s = s.replace('7', ' seven ')
        s = s.replace('8', ' eight ')
        s = s.replace('9', ' nine ')
        return  s
    
    def clean_text(self, s):
        if self.clean_text_v == 1:
            s = self.clean_text_v1(s)
        elif self.clean_text_v == 2:
            s = self.clean_text_v1(s)
            s = self.clean_text_v2(s)
        elif self.clean_text_v == 3:
            s = self.clean_text_v2(s)

        return s
     

    def model_env_init(self):   
        torch.set_num_threads(self.num_threads)
        torch.set_num_interop_threads(self.num_threads)
        
        if self.n_gpu > 0:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu  = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                              
        return None

    def model_init(self):
        confg_class, model_class, token_class = MODEL_CLASSES[self.params['algo']['model_type']]

        config         = confg_class.from_pretrained(self.params['algo']['model_name'], num_labels=self.params['num_class'], finetuning_task=self.params['objective'])
        self.tokenizer = token_class.from_pretrained(self.params['algo']['model_name'], do_lower_case=self.params['algo']['tokenizer_do_lower_case']) 
        ml_model       = model_class.from_pretrained(self.params['algo']['model_name'], from_tf=False, config=config)

        ml_model.to(self.device)

        return ml_model
    
    def model_params(self):
        self.params['algo']['model_type']                   = {transformer_model}

        model_name_id = 0
        n_model_names = len(MODEL_NAMES[self.params['algo']['model_type']])
        mappd_name_id = self.map_from_continuous_set_to_set(10, n_model_names, model_name_id)  # assuming 10 random options in "transformer_model_id" - change if expanded

        self.params['algo']['model_name']                   = MODEL_NAMES[self.params['algo']['model_type']][mappd_name_id]
        
        self.params['algo']['num_train_epochs']             = {epoch}
        self.params['algo']['max_train_steps']              = {max_train_steps}

        self.params['algo']['max_seq_length']               = {max_seq_length}
        self.params['algo']['tokenizer_do_lower_case']      = {tokenizer_do_lower_case}

        self.params['algo']['learning_rate']                = {learning_rate}
        self.params['algo']['gradient_accumulation_steps']  = {gradient_accumulation_steps}
        self.params['algo']['train_batch_size']             = {train_batch_size}
        self.params['algo']['weight_decay']                 = {weight_decay}
        self.params['algo']['adam_epsilon']                 = {adam_epsilon}
        self.params['algo']['warmup_ratio']                 = {warmup_ratio}
        self.params['algo']['warmup_steps']                 = {warmup_steps}
        self.params['algo']['max_grad_norm']                = {max_grad_norm}

        self.params['algo']['fp16']                         = {fp16}
        self.params['algo']['fp16_opt_level']               = {fp16_opt_level}

        self.params['algo']['logging_steps']                = {logging_steps}
        self.params['algo']['random_seed']                  = self.rn_seed_init

        self.params['algo']['thread_count']                 = {num_threads}
        self.params['algo']['verbose']                      = 1
        
        if self.is_binary:
            print ("detected binary target: use AUC/LOGLOSS and Binary Cross Entropy loss evaluation")
            self.params['objective']                     = 'binary'
            self.params['algo']['eval_metric']           = 'AUC'
            self.params['num_class']                     = 2
            # self.params['prediction_type']             = 'Probability'
            # self.params['loss_function']               = 'crossentropy'
            # self.params['metric']                      = [self.tf_roc_auc, self.tf_prc_auc]                                 # if using custom metric function cannot save in params as pickle will fail
            # self.metric                                = [self.tf_roc_auc, self.tf_prc_auc]                                 # in such case use local class variable for metric
            # self.params['early_stop_metric']           = 'val_tf_prc_auc'
            # self.params['early_stop_metric_direction'] = 'max'
        elif self.is_set(self.objective_multiclass):
            print ("detected multi-class target: use Multi-LogLoss/Error; " + str(len(self.target_classes)) + " classes")
            self.params['objective']                     = self.objective_multiclass
            self.params['algo']['eval_metric']           = 'MultiClassOneVsAll'
            self.params['num_class']                     = int(max(self.target_classes) + 1)  # requires all int numbers from 0 to max to be classes
            # self.params['prediction_type']             = 'Probability'
            # self.params['loss_function']               = 'MultiClassOneVsAll'
            # self.params['metric']                      = ['accuracy']
            # self.metric                                = ['accuracy']
            # self.params['early_stop_metric']           = 'val_loss'
            # self.params['early_stop_metric_direction'] = 'auto'
        else:
            print ("detected regression target: use RMSE/MAE")
            self.params['objective']                     = self.objective_regression
            self.params['algo']['eval_metric']           = 'RMSE'
            self.params['num_class']                     = 1
            # self.params['prediction_type']             = 'RawFormulaVal'
            # self.params['loss_function']               = 'RMSE'
            # self.params['metric']                      = ['mean_squared_error']
            # self.metric                                = ['mean_squared_error']
            # self.params['early_stop_metric']           = 'val_loss'
            # self.params['early_stop_metric_direction'] = 'auto'           
            # params['metric']                           = ['rmse', 'mae']


    def model_predict(self, ml_model, xt):
        try:
            dataset_eval    = self.convert_df_to_dataset(xt, None, self.tokenizer)
            eval_batch_size = len(dataset_eval)

            eval_sampler    = SequentialSampler(dataset_eval)
            eval_dataloader = DataLoader(dataset_eval, sampler=eval_sampler, batch_size=eval_batch_size)

            self.logger.info("***** Running prediction *****")
            self.logger.info("  Num examples = %d", len(dataset_eval))
            self.logger.info("  Batch size   = %d", eval_batch_size)

            pred = None

            for batch in tqdm(eval_dataloader, desc="Predicting"):
                ml_model.eval()
                batch = tuple(t.to(self.device) for t in batch)

                with torch.no_grad():
                    inputs = {'input_ids'     : batch[0],
                              'attention_mask': batch[1]}

                    if self.params['algo']['model_type'] != 'distilbert':
                        inputs['token_type_ids'] = batch[2] if self.params['algo']['model_type'] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                    outputs = ml_model(**inputs)
                    logits  = outputs[0]

                    if pred is None:
                        pred = softmax(logits.detach().cpu().numpy())
                    else:
                        pred = np.append(pred, softmax(logits.detach().cpu().numpy()), axis=0)

            if self.is_binary:
                pred = np.array( [r[1] for r in pred] )
            elif self.params['objective'] == self.objective_regression:
                pred = np.squeeze(pred)

        except Exception as e:
            print ('Transformer Predict error: ', e)
            pred = 0

        return pred

    def model_save(self, predictor, file_path):
        if not os.path.exists(file_path):
            os.makedirs(file_path)
                
        predictor.save_pretrained(file_path)
        return

    def model_load(self, file_path):
        confg_class, model_class, token_class = MODEL_CLASSES[self.dicts_agent['params']['algo']['model_type']]
        ml_model = model_class.from_pretrained(file_path)

        ml_model.to(self.device)
        
        return ml_model


    def convert_example_to_features( self, example_text, tokenizer, 
                                     max_seq_length=128, pad_on_left=False, mask_padding_with_zero=True,
                                     pad_token=0, pad_token_segment_id=0, 
                                     sequence_a_segment_id = 0,
                                     cls_token_at_end=False,
                                     cls_token='<cls>',
                                     cls_token_segment_id  = 0, 
                                     sep_token='<sep>', sep_token_extra=False):

        example_tokens = tokenizer.tokenize(example_text)
        self.example_tokens_pub=example_tokens

        # Account for [CLS] and [SEP] with "- 2" or "- 3" for RoBERTa.
        special_tokens_count = 3 if sep_token_extra else 2
        if len(example_tokens) > max_seq_length - special_tokens_count:
            example_tokens = example_tokens[:(max_seq_length - special_tokens_count)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids:   0   0   0   0  0     0   0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens      = example_tokens + [sep_token]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        if cls_token_at_end:
            tokens      = tokens      + [cls_token]
            segment_ids = segment_ids + [cls_token_segment_id]
        else:
            tokens =      [cls_token]            + tokens
            segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length  = max_seq_length - len(input_ids)
        if pad_on_left:
            input_ids   = ([pad_token] * padding_length) + input_ids
            input_mask  = ([0 if mask_padding_with_zero else 1] * padding_length) + input_mask
            segment_ids = ([pad_token_segment_id] * padding_length) + segment_ids
        else:
            input_ids   = input_ids + ([pad_token] * padding_length)
            input_mask  = input_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            segment_ids = segment_ids + ([pad_token_segment_id] * padding_length)

        assert len(input_ids)   == max_seq_length
        assert len(input_mask)  == max_seq_length
        assert len(segment_ids) == max_seq_length

        return (input_ids, input_mask, segment_ids)


    def convert_df_to_dataset(self, x_data, y_data, tokenizer):
        all_input_ids   = []
        all_input_mask  = []
        all_segment_ids = []

        pad_on_left          = bool(self.params['algo']['model_type'] in ['xlnet'])
        pad_token            = tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0]
        pad_token_segment_id = 4 if self.params['algo']['model_type'] in ['xlnet'] else 0

        cls_token_at_end     = bool(self.params['algo']['model_type'] in ['xlnet'])
        cls_token            = tokenizer.cls_token
        cls_token_segment_id = 2 if self.params['algo']['model_type'] in ['xlnet'] else 0

        sep_token            = tokenizer.sep_token
        sep_token_extra      = bool(self.params['algo']['model_type'] in ['roberta'])

        for rowTuple in x_data.itertuples(index=False):
            row = ''
            for col in rowTuple:
                row += ' ' + str(col)

            row = row[1:]

            input_ids, input_mask, segment_ids = self.convert_example_to_features(row, tokenizer,
                                                        max_seq_length         = self.params['algo']['max_seq_length'], 
                                                        pad_on_left            = pad_on_left,
                                                        mask_padding_with_zero = True,
                                                        pad_token              = pad_token, 
                                                        pad_token_segment_id   = pad_token_segment_id,
                                                        cls_token_at_end       = cls_token_at_end,
                                                        cls_token              = cls_token,
                                                        cls_token_segment_id   = cls_token_segment_id,
                                                        sep_token              = sep_token,
                                                        sep_token_extra        = sep_token_extra )

            all_input_ids.append(input_ids)
            all_input_mask.append(input_mask)
            all_segment_ids.append(segment_ids)

        all_input_ids   = torch.tensor(all_input_ids,    dtype=torch.long)
        all_input_mask  = torch.tensor(all_input_mask,   dtype=torch.long)
        all_segment_ids = torch.tensor(all_segment_ids,  dtype=torch.long)

        if y_data is not None:
            all_label_ids = torch.tensor(y_data.to_list(), dtype=torch.long)
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
        else:
            dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids)

        return dataset


    def model_train(self, ml_model, x_train, y_train, x_test, y_test, current_fold):

        dataset_train    = self.convert_df_to_dataset(x_train, y_train, self.tokenizer)

        train_sampler    = RandomSampler(dataset_train)
        train_dataloader = DataLoader(dataset_train, sampler=train_sampler, batch_size=self.params['algo']['train_batch_size'])

        if self.params['algo']['max_train_steps'] > 0:
            t_total = self.params['algo']['max_train_steps']
            self.params['algo']['num_train_epochs'] = self.params['algo']['max_train_steps'] // (len(train_dataloader) // self.params['algo']['gradient_accumulation_steps']) + 1
        else:
            t_total = len(train_dataloader) // self.params['algo']['gradient_accumulation_steps'] * self.params['algo']['num_train_epochs']

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in ml_model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.params['algo']['weight_decay']},
            {'params': [p for n, p in ml_model.named_parameters() if     any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        warmup_steps = math.ceil(t_total * self.params['algo']['warmup_ratio'])
        self.params['algo']['warmup_steps'] = warmup_steps if self.params['algo']['warmup_steps'] == 0 else self.params['algo']['warmup_steps']

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.params['algo']['learning_rate'], eps=self.params['algo']['adam_epsilon'])
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=self.params['algo']['warmup_steps'], t_total=t_total)

        if self.params['algo']['fp16']:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            ml_model, optimizer = amp.initialize(ml_model, optimizer, opt_level=self.params['algo']['fp16_opt_level'])

        self.logger.info("***** Running training *****")
        self.logger.info("  Num examples = %d",                len(dataset_train))
        self.logger.info("  Num Epochs   = %d",                self.params['algo']['num_train_epochs'])
        self.logger.info("  Total train batch size      = %d", self.params['algo']['train_batch_size'] * self.params['algo']['gradient_accumulation_steps'])
        self.logger.info("  Gradient Accumulation steps = %d", self.params['algo']['gradient_accumulation_steps'])
        self.logger.info("  Total optimization steps    = %d", t_total)


        global_step  = 0
        tr_loss      = 0.0 
        logging_loss = 0.0

        ml_model.zero_grad()
        train_iterator = trange(int(self.params['algo']['num_train_epochs']), desc="Epoch")

        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                ml_model.train()
                batch  = tuple(t.to(self.device) for t in batch)

                inputs = {'input_ids':      batch[0],
                          'attention_mask': batch[1],  
                          'labels':         batch[3]}

                if self.params['algo']['model_type'] != 'distilbert':
                    inputs['token_type_ids'] = batch[2] if self.params['algo']['model_type'] in ['bert', 'xlnet'] else None  # XLM, DistilBERT and RoBERTa don't use segment_ids

                outputs = ml_model(**inputs)
                loss    = outputs[0]  # model outputs are always tuple in transformers (see doc)

                if self.params['algo']['gradient_accumulation_steps'] > 1:
                    loss = loss / self.params['algo']['gradient_accumulation_steps']

                if self.params['algo']['fp16']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), self.params['algo']['max_grad_norm'])					
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(ml_model.parameters(), self.params['algo']['max_grad_norm'])

                tr_loss += loss.item()
                if (step + 1) % self.params['algo']['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()      # Update learning rate schedule
                    ml_model.zero_grad()
                    global_step += 1

                    if self.params['algo']['logging_steps'] > 0 and global_step % self.params['algo']['logging_steps'] == 0:
                        # Log metrics
                        self.logger.info('\n')
                        self.logger.info(' global_step = %s, lrate = %s', global_step, scheduler.get_lr()[0])
                        self.logger.info(' global_step = %s, loss  = %s', global_step, (tr_loss - logging_loss)/self.params['algo']['logging_steps'])
                        logging_loss = tr_loss

                if self.params['algo']['max_train_steps'] > 0 and global_step > self.params['algo']['max_train_steps']:
                    epoch_iterator.close()
                    break

            if self.params['algo']['max_train_steps'] > 0 and global_step > self.params['algo']['max_train_steps']:
                train_iterator.close()
                break

        tr_loss = tr_loss / global_step
        self.logger.info(" global_step = %s, average loss = %s", global_step, tr_loss)

        return ml_model

                    
    def load_columns(self, map_dict=True):
        # start from loading the target field
        df_all = pd.read_csv(workdir+self.target_file, usecols=[self.target_col])[[self.target_col]]

        columns_new = [self.target_col]
        columns     = [self.target_col]
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        print (str(datetime.now()), " start loading data")
        block_progress = 0
        block          = int(self.fields_to_use/20)

        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]

            if self.is_use_column(col_name):
                df_col = pd.read_csv(workdir+file_name, usecols=[col_name])[[col_name]]       # read column from csv file
                
                # if column has associated dictionary csv then it's a text column, replace column with actual text
                dict_file_name = workdir+'dict_'+col_name+'.csv'
                if os.path.isfile(dict_file_name) and map_dict:
                    dict1 = pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()       # load dictionary
                    df_col[col_name] = df_col[col_name].map(dict1)                                                         # map and replace
                    self.dicts_agent[col_name] = dict1                                                                     # save in dictionary of dictionaries to be saved with model files
                    
                    df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text)
                else:
                    if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:           # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(np.float32)

                df_all = df_all.merge(df_col, left_index=True, right_index=True)                       # add column to the overall dataframe

                block_progress += 1
                if (block_progress >= block):
                    block_progress = 0
                    print (str(datetime.now()), " data loaded: ", round((i+1)/self.fields_to_use*100,0), "%")

                # some columns may appear multiple times in data_defs as inherited from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count==1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name+"_v"+str(ncol_count))

        # rename columns in df to unique names
        df_all.columns = columns_new
        print (str(datetime.now()), " data loaded", len(df_all), "rows; ", len(df_all.columns), "columns")
        return df_all
        
        
    def apply(self, df_add):
        # this method is called by AIOS when additional data is supplied and needs to be predicted on
        # df_add shouldn't contain columns with text values - only numeric
        global dicts
        # by this stage all text fields should have been converted to dictionary values by previous agents that created such fields in AIOS
        # since this agent works with text fields, actual text values will be obtained from the global dicts variable
        columns_new = []
        columns     = []
        # assemble a list of column names given to the agent by AIOS in (data) DNA gene up-to (fields_to_use) gene
        for i in range(0,self.fields_to_use):
            col_name  = self.data_defs[i].split("|")[0]
            file_name = self.data_defs[i].split("|")[1]
            
            if self.is_use_column(col_name):               
                # assemble dataframe column by column             
                df_col = df_add[[col_name]]
                              
#                 # if column has associated dictionary csv then it's a text column, replace column with actual text
#                 dict_file_name = workdir+'dict_'+col_name+'.csv'
#                 if self.os.path.isfile(dict_file_name) and self.map_dict:
#                     dict1 = pd.read_csv(dict_file_name, dtype={'value': object}).set_index('key')["value"].to_dict()  # load dictionary
#                     df_col[col_name] = df_col[col_name].map(dict1)                                                    # map and replace
                
                # if column is in global dictionary then it's a text column, replace column with actual text
                if col_name in dicts:
                    reverse_dict1    = {v:k for k,v in dicts[col_name].items()}
                    df_col[col_name] = df_col[col_name].map(reverse_dict1)                   
                    df_col[col_name] = df_col[col_name].astype(str).apply(self.clean_text)    
                else:
                    if df_col[col_name].dtype == np.float64 and self.use_float32_dtype:                               # downcast to save memory if needed
                        df_col[col_name] = df_col[col_name].astype(np.float32)
                        
                if i==0:
                    df = df_col[[col_name]]
                else:
                    df = df.merge(df_col[[col_name]], left_index=True, right_index=True)
                
                # some columns may appear multiple times in data_defs as inhereted from parents DNA
                # assemble a list of columns assigning unique names to repeating columns
                columns.append(col_name)
                ncol_count = columns.count(col_name)
                if ncol_count==1:
                    columns_new.append(col_name)
                else:
                    columns_new.append(col_name+"_v"+str(ncol_count))
        
        # rename columns in df to unique names
        df.columns = columns_new
        
        # predict new data set in df applying model for each fold used for training
        pred = np.zeros(len(df))
        if self.dicts_agent['params']['objective'] == self.objective_multiclass:
            # create a list of lists depending on number of classes used for training 
            # as each prediction is a list of values against each class
            pred = [np.zeros(self.dicts_agent['params']['num_class']) for i in range(len(df))]
         
        # apply model from each fold created during training and sum their predictions
        if self.dicts_agent['params']['random_folds'] == False:
            for fold in range(self.start_fold, self.nfolds):
                pred += self.model_predict(predictors[fold - self.start_fold], df)

            if self.dicts_agent['params']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / (self.nfolds - self.start_fold)
        else:
            for fold in range(0, len(self.predictors)):
                pred += self.model_predict(predictors[fold], df)

            if self.dicts_agent['params']['objective'] == self.objective_multiclass:
                # select class with largest total value in case of multiclass
                pred = np.argmax(pred, axis=1)
            else:
                # average prediction over all folds in case of binary or regression
                pred = pred / len(self.predictors)
        
        df_add[self.output_column] = pred
        

    def run(self, mode):
        # this is main method called by AIOS with supplied DNA Genes to process data
        print ("enter run mode " + str(mode))  # 0=work for fitness only;  1=make new output field

        # prepare all parameters
        self.params['random_valid']       = {random_valid}
        self.params['random_valid_size']  = {random_valid_size}
        self.params['random_valid_folds'] = {random_valid_folds}
        self.params['random_folds']       = {random_folds}
        self.params['random_folds_size']  = {random_folds_size}
        self.params['binary_balancing']   = {binary_balancing}
        self.params['binary_balancing_0'] = {binary_balancing_0}
        self.params['binary_balancing_1'] = {binary_balancing_1}
        self.params['binary_eval_fun']    = {binary_eval_fun}

        # obtain indexes for train and remainder sets
        # load target column as it may be needed for filtering and removing NaN targets from training
        df_filter_column       = pd.read_csv(workdir + self.target_file, usecols=[self.target_col])
        filter_condition_train = df_filter_column[self.target_col].notnull()

        # applying specified filters
        if self.is_set(self.filter_column):
            # load columns to filter by
            df_t = pd.read_csv(workdir + self.filter_filename, usecols=[self.filter_column])
            df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)

            filter_condition_train = np.logical_and( filter_condition_train,
                                        np.logical_and( df_filter_column[self.filter_column] >= {train_set_from},
                                                             df_filter_column[self.filter_column] <  {train_set_to} ) )

            # two filter columns specified
            if self.is_set(self.filter_column_2):
                df_t = pd.read_csv(workdir + self.filter_filename_2, usecols=[self.filter_column_2])
                df_filter_column = df_filter_column.merge(df_t, left_index=True, right_index=True)

                condition2 = np.logical_and(df_filter_column[self.filter_column_2] >= {train_set_from_2},
                                                 df_filter_column[self.filter_column_2] <  {train_set_to_2} )
                filter_condition_train = np.logical_and(filter_condition_train, condition2)

        train_filtered_indexes = df_filter_column[filter_condition_train].index.tolist()
        remainder_set_indexes  = df_filter_column[np.logical_not(filter_condition_train)].index.tolist()  # remainder which is not in train

        # initialise prediction column for entire data set as it will be aggregate prediction from multiple folds
        df_filter_column[self.output_column + '_folds_pred']       = 0
        df_filter_column[self.output_column + '_folds_pred_count'] = 0  # number of predictions for each record as different folds will predict different records, so each record may have unique number of predictions

        # load specified in data_defs colums of data up-to fields_to_use quantity
        df_all = self.load_columns(map_dict=self.map_dict)
        original_row_count = len(df_all)

        # analyse target column whether it is binary which may result in different loss function used
        self.target_classes = df_all[df_all[self.target_col].notnull()].sort_values(self.target_col)[self.target_col].unique().tolist()
        self.is_binary      = self.target_classes==[0, 1]

        self.params['input_dim'] = len(df_all.columns) - 1  # need this for some models init; df.columns includes the target column, hence need to do -1

        # configure ML model specific parameters which will be saved in self.params dictionary
        self.model_params()

        # initialise temp df holding multi-class predictions for entire data set
        df_filter_column_mc = pd.DataFrame([np.zeros(self.params['num_class']) for i in range(len(df_filter_column))])

        self.dicts_agent['params']   = self.params

        train_sets_ix                = []      # indexes of each whole set used for training
        valid_sets_ix                = []
        train_sub_sets_ix            = []      # indexes of each subset of whole set used for training
        test_sub_sets_ix             = []      # indexes of each subset of whole set used for out-of-sample testing during training
        predictors_all               = []
        weighted_result_folds        = []
        weighted_auc_folds           = []
        valid_result_folds           = []
        valid_result_auc_folds       = []

        fold_all = 0
        # repeat cross-validation multiple times with different validation set each time
        # applies only in case when params['random_valid'] == True
        for valid_fold in range(0, self.params['random_valid_folds']):
            print ()
            print (str(datetime.now()), " ----- VALID FOLD: ", valid_fold)
            # obtain indexes for validation set if required
            # applying specified filters
            if self.use_validation_set:
                # assemble condition for filtering validation set
                filter_condition_valid = df_filter_column[self.target_col].notnull()

                if self.is_set(self.filter_column):
                    filter_condition_valid = np.logical_and(filter_condition_valid,
                                                                 np.logical_and( df_filter_column[self.filter_column] >= {valid_set_from},
                                                                                      df_filter_column[self.filter_column] <  {valid_set_to} ) )
                    # two filter columns specified
                    if self.is_set(self.filter_column_2):
                        condition2 = np.logical_and( df_filter_column[self.filter_column_2] >= {valid_set_from_2},
                                                          df_filter_column[self.filter_column_2] <  {valid_set_to_2} )
                        filter_condition_valid = np.logical_and(filter_condition_valid, condition2)

                if self.params['random_valid'] == False:
                    # select validation based on fixed filter - may intersect with test or remainder set
                    train_sets_ix.append(train_filtered_indexes)
                    valid_sets_ix.append(df_filter_column[filter_condition_valid].index.tolist())
                else:
                    # apply stratified random selection to previously filtered train set
                    sss = StratifiedShuffleSplit(n_splits=1, test_size=self.params['random_valid_size'])
                    y   = df_filter_column[df_filter_column.index.isin(train_filtered_indexes)][[self.target_col]]
                    iy  = y.reset_index(level=0)                                              # create copy, save existing index in 'index' column and reset index
                    y.reset_index(drop=True, inplace=True)                                    # reset index because StratifiedShuffleSplit will reset index anyway

                    for train_ix, valid_ix in sss.split(np.zeros(len(y)), y):
                        train_sets_ix.append( iy[iy.index.isin(train_ix)]['index'].tolist())  # obtain original indexes from saved copy of labels with original indexes
                        valid_sets_ix.append( iy[iy.index.isin(valid_ix)]['index'].tolist())  # can't use train_ix, valid_ix directly because they refer to new index reset during shuffling
                        print ('TRAIN target mean: ', df_filter_column[df_filter_column.index.isin(train_sets_ix[valid_fold])][self.target_col].mean().round(3))
                        print ('VALID target mean: ', df_filter_column[df_filter_column.index.isin(valid_sets_ix[valid_fold])][self.target_col].mean().round(3))

            # save indexes used for splits
            self.dicts_agent['train_sets_ix']    = train_sets_ix
            self.dicts_agent['remainder_set_ix'] = remainder_set_indexes
            self.dicts_agent['valid_sets_ix']    = valid_sets_ix

            print ("Length of train set:",          len(train_sets_ix[valid_fold]))
            print ("Length of test/remainder set:", len(remainder_set_indexes))
            print ("Length of validation set:",     len(valid_sets_ix[valid_fold]))

            # duplicate originally loaded data
            df      = df_all.copy()
            # use previously calculated indexes to select train, validation and remainder sets
            df_test = df[df.index.isin(remainder_set_indexes)]

            if self.use_validation_set:
                df_valid = df[df.index.isin(valid_sets_ix[valid_fold])]
                # initialise prediction column for validation as it will be aggregate prediction from multiple folds
                predicted_valid_set = np.zeros(len(df_valid))
                # Multi-class case: initialise prediction list of lists depending on number of classes
                # as each prediction is a list of values against each class
                if self.params['objective'] == self.objective_multiclass:
                    predicted_valid_set = [np.zeros(self.params['num_class']) for i in range(len(df_valid))]

            df = df[df.index.isin(train_sets_ix[valid_fold])]

            # initialise prediction column for main train set as it will be aggregate prediction from multiple folds
            prediction         = np.zeros(len(df))
            # initialise prediction column for remainder set as it will be aggregate prediction from multiple folds
            predicted_test_set = np.zeros(len(df_test))
            # Multi-class case: initialise prediction list of lists depending on number of classes
            # as each prediction is a list of values against each class
            if self.params['objective'] == self.objective_multiclass:
                prediction         = [np.zeros(self.params['num_class']) for i in range(len(df))]
                predicted_test_set = [np.zeros(self.params['num_class']) for i in range(len(df_test))]

            #############################################################
            #                   MAIN LOOP
            #############################################################

            weighted_result = 0
            weighted_auc    = 0
            count_records_notnull = 0

            if self.params['random_folds'] == False:
                # divide training data into nfolds of size block
                block = int(len(df) / self.nfolds)
                # select folds sequentially in existing index order
                for fold in range(self.start_fold, self.nfolds):
                    print ()
                    print (str(datetime.now()), " Train/Test FOLD: ", fold)
                    range_start = fold*block
                    range_end   = (fold+1)*block
                    if fold==self.nfolds-1:
                        range_end = len(df)
                    range_predict = range(range_start, range_end)
                    print ("Fold to predict start", range_start, "; end ", range_end)

                    x_test = df[df.index.isin(range_predict)]
                    x_test.reset_index(drop=True, inplace=True)
                    x_test_orig = x_test.copy()                                 # save original test set before removing null values
                    x_test = x_test[x_test[self.target_col].notnull()]          # remove examples that have no proper target label
                    x_test.reset_index(drop=True, inplace=True)

                    x_train = df[df.index.isin(range_predict)==False]
                    x_train.reset_index(drop=True, inplace=True)
                    x_train = x_train[x_train[self.target_col].notnull()]       # remove examples that have no proper target label
                    x_train.reset_index(drop=True, inplace=True)

                    print ("x_test rows count: " + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))

                    y_train = x_train[self.target_col]          # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)

                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold-self.start_fold+1)
                    pred      = self.model_predict(predictor, x_test)

                    if mode==1:
                        self.model_save(predictor, workdir + self.output_column + "_fold" + str(fold) + ".model")

                    if self.is_binary:
                        result = my_log_loss(y_test, pred)
                        # show various metrics as per
                        # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                        result_roc_auc = roc_auc_score(y_test, pred)
                        result_prc_auc = self.prc_auc(y_test, pred)
                        print ("ROC AUC score: ", result_roc_auc)
                        print ("PRC AUC score: ", result_prc_auc)

                        if self.print_tables:
                            result_cm = confusion_matrix(y_test, (pred > 0.5))  # assume 0.5 probability threshold
                            result_cr = classification_report(y_test, (pred > 0.5))
                            print ("Confusion Matrix:\n", result_cm)
                            print ("Classification Report:\n", result_cr)
                    elif self.is_set(self.objective_multiclass):
                        pred_classes = np.argmax(pred, axis=1)
                        result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                        result_acc_score  = accuracy_score(y_test, pred_classes)
                        result_cm = confusion_matrix(y_test, pred_classes)
                        result_cr = classification_report(y_test, pred_classes)
                        if self.print_tables:
                            print ("Precision score: ", result_prec_score)
                            print ("Accuracy score: ", result_acc_score)
                            print ("Confusion Matrix:\n", result_cm)
                            print ("Classification Report:\n", result_cr)
                        result = predictor.best_score['valid_0']['multi_logloss']
                        result_roc_auc = f1_score(y_test, pred_classes, average='weighted')
                    else:
                        result = sum(abs(y_test - pred)) / len(y_test)
                        # result = sqrt(mean_squared_error(y_test, pred))

                    print ("result: ", result)

                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    # predict all examples in the original test set which may include erroneous examples previously removed
                    pred_all_test = self.model_predict(predictor, x_test_orig.drop(self.target_col, axis=1))

                    if self.params['objective'] == self.objective_multiclass:
                        prediction[range_start:range_end] = np.argmax(pred_all_test, axis=1)
                    else:
                        prediction[range_start:range_end] = pred_all_test

                    # predict validation and remainder sets examples
                    if self.use_validation_set:
                        predicted_valid_set += self.model_predict(predictor, df_valid.drop(self.target_col, axis=1))
                        predicted_test_set  += self.model_predict(predictor, df_test.drop(self.target_col, axis=1))

                predicted_valid_set = predicted_valid_set / (self.nfolds - self.start_fold)
                predicted_test_set  = predicted_test_set / (self.nfolds - self.start_fold)
            else:
                # select folds using random shuffle and stratify
                sss = StratifiedShuffleSplit(n_splits=self.nfolds, test_size=self.params['random_folds_size'])
                y   = df[[self.target_col]]
                iy  = y.reset_index(level=0)            # create copy, save existing index in 'index' column and reset index
                y.reset_index(drop=True, inplace=True)  # reset index because StratifiedShuffleSplit will reset index anyway

                predictors = []
                for train_ix, test_ix in sss.split(np.zeros(len(y)), y):
                    fold_all += 1
                    print ()
                    print (str(datetime.now()), " Train/Test FOLD: ", fold_all)

                    train_ix_orig = iy[iy.index.isin(train_ix)]['index'].tolist()  # obtain original indexes from saved copy of labels with original indexes
                    test_ix_orig  = iy[iy.index.isin(test_ix)]['index'].tolist()   # can't use train_ix, test_ix directly because they refer to new index reset during shuffling

                    # ------ balance train set -----------------------------------------------------------------------------------------------------
                    if self.params['binary_balancing']:
                        bal_y = df[[self.target_col]]
                        # under sample both binary label samples by fixed per label percentage
                        bal_cond = np.logical_and(bal_y.index.isin(train_ix_orig), bal_y[self.target_col] == 0)
                        train_ix_orig_balanced_0 = bal_y[bal_cond].index.tolist()
                        train_balanced_size_0    = int(len(train_ix_orig_balanced_0) * self.params['binary_balancing_0'])
                        train_ix_orig_balanced_0 = np.random.choice(train_ix_orig_balanced_0, train_balanced_size_0, replace=False).tolist()

                        bal_cond = np.logical_and(bal_y.index.isin(train_ix_orig), bal_y[self.target_col] == 1)
                        train_ix_orig_balanced_1 = bal_y[bal_cond].index.tolist()
                        train_balanced_size_1    = int(len(train_ix_orig_balanced_1) * self.params['binary_balancing_1'])
                        train_ix_orig_balanced_1 = np.random.choice(train_ix_orig_balanced_1, train_balanced_size_1, replace=False).tolist()

                        train_ix_orig = train_ix_orig_balanced_0 + train_ix_orig_balanced_1
                    # ------------------------------------------------------------------------------------------------------------------------------

                    train_sub_sets_ix.append(train_ix_orig)  # save indexes in the overall list for all folds
                    test_sub_sets_ix.append(test_ix_orig)

                    x_test  = df[df.index.isin(test_ix_orig)]
                    x_train = df[df.index.isin(train_ix_orig)]

                    print ("x_test  rows count: " + str(len(x_test)))
                    print ("x_train rows count: " + str(len(x_train)))

                    y_train = x_train[self.target_col]  # separate training fields and the target
                    x_train = x_train.drop(self.target_col, 1)

                    y_test = x_test[self.target_col]
                    x_test = x_test.drop(self.target_col, 1)

                    print ('Y_TEST  Target mean: ', y_test.mean().round(3))
                    print ('Y_TRAIN Target mean: ', y_train.mean().round(3))

                    predictor = self.model_init()
                    predictor = self.model_train(predictor, x_train, y_train, x_test, y_test, fold_all)
                    pred      = self.model_predict(predictor, x_test)

                    try:
                        if self.is_binary:
                            result = log_loss(y_test, pred)
                            # show various metrics as per
                            # http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report
                            result_roc_auc = roc_auc_score(y_test, pred)
                            result_prc_auc = self.prc_auc(y_test, pred)
                            print ("ROC AUC score: ", result_roc_auc)
                            print ("PRC AUC score: ", result_prc_auc)

                            if self.print_tables:
                                result_cm = confusion_matrix(y_test, np.asarray(pred) > 0.5)  # assume 0.5 probability threshold
                                result_cr = classification_report(y_test, np.asarray(pred) > 0.5)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1

                        elif self.params['objective'] == self.objective_multiclass:
                            pred_classes = np.argmax(pred, axis=1)
                            result_prec_score = precision_score(y_test, pred_classes, average='weighted')
                            result_acc_score  = accuracy_score(y_test, pred_classes)
                            result_cm = confusion_matrix(y_test, pred_classes)
                            result_cr = classification_report(y_test, pred_classes)

                            if self.print_tables:
                                print ("Precision score: ", result_prec_score)
                                print ("Accuracy score: ",  result_acc_score)
                                print ("Confusion Matrix:\n",      result_cm)
                                print ("Classification Report:\n", result_cr)

                            result         = log_loss(y_test, pred)
                            result_roc_auc = f1_score(y_test, pred_classes, average='weighted')

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_pred = np.array(df_filter_column_mc.loc[test_ix_orig])  # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[test_ix_orig] = df_pred                 # temp df holding multi-class prediction
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1

                        else:
                            result = sum(abs(y_test - pred)) / len(y_test)
                            # result = sqrt(mean_squared_error(y_test, pred))

                            # assign predictions to corresponding test records only
                            # this is done to make sure predicted labels are always out of sample i.e., avoiding leaks in model stacking
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[test_ix_orig, self.output_column + '_folds_pred_count'] += 1
                    except Exception as e:
                        print (e)
                        result = 999999
                        result_roc_auc = 0

                    print ("result: ", result)

                    weighted_result += result * len(pred)
                    weighted_auc    += result_roc_auc * len(pred)
                    count_records_notnull += len(pred)

                    if result_roc_auc < self.min_perf_criteria:
                        print ("Minimum performance criteria: " + str(self.min_perf_criteria) + " not met! result_roc_auc: " + str(result_roc_auc))
                        return

                    if self.params['binary_eval_fun'] == 'PRCAUC':
                        predictors.append([predictor, result, result_prc_auc])
                        predictors_all.append([predictor, result, result_prc_auc])    # add predictors to global list across all validation folds
                    else:
                        predictors.append([predictor, result, result_roc_auc])
                        predictors_all.append([predictor, result, result_roc_auc])    # add predictors to global list across all validation folds

                #-------------- end of train test CV loop ---------------------------------------------------------------------------------------------

                predictors = pd.DataFrame(predictors, columns=['predictor', 'result', 'result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
                print ('\nFolds Performance Overall:')
                self.print_html(predictors, max_rows=50, max_cols=5)

                # select 3 predictors (best, worst and average) to be used for predicting all validation and remaining samples
                predictors['result_roc_auc_mean']      = predictors['result_roc_auc'].mean()
                predictors['result_roc_auc_mean_diff'] = abs(predictors['result_roc_auc'] - predictors['result_roc_auc_mean'])

                best_predictor_idx  = predictors['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors['result_roc_auc_mean_diff'].idxmin()

                predictors = [ predictors['predictor'][worst_predictor_idx],
                               predictors['predictor'][avg_predictor_idx],
                               predictors['predictor'][best_predictor_idx] ]

                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])

                #------------------ predict remaining and validation samples --------------------------------------------
                for fold in range(0, len(predictors)):
                    # predict remainder in the column output mode
                    if len(df_test) > 0 and mode == 1:
                        pred = self.model_predict(predictors[fold], df_test.drop(self.target_col, axis=1))
                        predicted_test_set += pred

                        if self.params['objective'] == self.objective_multiclass:
                            # assign predictions to corresponding test records only
                            df_pred = np.array(df_filter_column_mc.loc[remainder_set_indexes])      # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[remainder_set_indexes] = df_pred                     # temp df holding multi-class prediction
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred_count'] += 1
                        else:
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[remainder_set_indexes, self.output_column + '_folds_pred_count'] += 1

                    # predict validation set
                    if self.use_validation_set:
                        df_valid_x = df_valid.drop(self.target_col, axis=1)
                        pred = self.model_predict(predictors[fold], df_valid_x)
                        predicted_valid_set += pred

                        if self.params['objective'] == self.objective_multiclass:
                            # assign predictions to corresponding test records only
                            df_pred = np.array(df_filter_column_mc.loc[valid_sets_ix[valid_fold]])  # get array of previous folds test records predictions
                            df_pred += pred
                            df_filter_column_mc.loc[valid_sets_ix[valid_fold]] = df_pred                 # temp df holding multi-class prediction
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred_count'] += 1
                        else:
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred']       += pred
                            df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column + '_folds_pred_count'] += 1

                        #if fold == 0:
                        #    valid_set_shap_values = shap.TreeExplainer(predictors[fold]).shap_values(df_valid_x)
                        #else:
                        #    valid_set_shap_values += shap.TreeExplainer(predictors[fold]).shap_values(df_valid_x)
                # ------------------ end of predicting remaining and validation samples ---------------------------------

                prediction = prediction / len(predictors)
                predicted_test_set  = predicted_test_set / len(predictors)
                predicted_valid_set = predicted_valid_set / len(predictors)

                df_filter_column[self.output_column + '_folds_pred_avg'] = df_filter_column[self.output_column + '_folds_pred'] / df_filter_column[self.output_column + '_folds_pred_count']
            #------------ end of train test CV method selection ---------------------------------------------------------

            weighted_result = weighted_result/count_records_notnull
            weighted_auc    = weighted_auc/count_records_notnull

            weighted_result_folds.append(weighted_result)
            weighted_auc_folds.append(weighted_auc)

            print ("\nweighted_result: ", weighted_result)
            print ("weighted_auc: ",      weighted_auc)

            # if multiclass convert list of lists into list of predicted labels
            if self.params['objective'] == self.objective_multiclass:
                predicted_valid_set = np.argmax(predicted_valid_set, axis=1)
                predicted_test_set  = np.argmax(predicted_test_set, axis=1)

            if self.use_validation_set:
                print()
                print ("*************  VALIDATION SET RESULTS  *****************")
                print ("Length of validation set:", len(predicted_valid_set))

                # validation set may have missing labels (NAN), for metrics calc find subset with proper labels
                df_valid['predicted_valid_set'] = predicted_valid_set
                df_valid = df_valid[df_valid[self.target_col].notnull()]
                #df_valid.reset_index(drop=True, inplace=True)
                y_valid             = np.array(df_valid[self.target_col])
                predicted_valid_set = np.array(df_valid['predicted_valid_set'])

                try:
                    if self.is_binary:
                        result = log_loss(y_valid, predicted_valid_set)
                        print ("LOGLOSS: ", result)
                        result_roc_auc = roc_auc_score(y_valid, predicted_valid_set)
                        print ("ROC AUC score: ", result_roc_auc)
                        result_prc_auc = self.prc_auc(y_valid, predicted_valid_set)
                        print ("PRC AUC score: ", result_prc_auc)

                        if self.print_tables:
                            result_cm = confusion_matrix(y_valid, (predicted_valid_set > 0.5))  # assume 0.5 probability threshold
                            print ("Confusion Matrix:\n",      result_cm)
                            result_cr = classification_report(y_valid, (predicted_valid_set > 0.5))
                            print ("Classification Report:\n", result_cr)

                        valid_result_folds.append(result)
                        valid_result_auc_folds.append(result_roc_auc)

                    elif self.params['objective'] == self.objective_multiclass:
                        result_prec_score = precision_score(y_valid, predicted_valid_set, average='weighted')
                        result_acc_score  = accuracy_score(y_valid, predicted_valid_set)
                        result_cm = confusion_matrix(y_valid, predicted_valid_set)
                        result_cr = classification_report(y_valid, predicted_valid_set)

                        if self.print_tables:
                            print ("Precision score: ",        result_prec_score)
                            print ("Accuracy score: ",         result_acc_score)
                            print ("Confusion Matrix:\n",      result_cm)
                            print ("Classification Report:\n", result_cr)

                        result = 1 - result_prec_score
                        result_roc_auc = f1_score(y_valid, predicted_valid_set, average='weighted')

                    else:
                        # result = sum(abs(y_valid-predicted_valid_set))/len(y_valid)
                        # print ("MAE: ", result)
                        result = sqrt(mean_squared_error(y_valid, predicted_valid_set))
                        valid_result_folds.append(result)
                        print ("Root Mean Squared Error: ", result)
                except Exception as e:
                    print (e)
                    return  # no point to carry on with more folds

                print ("\n************* END of VALIDATION SET RESULTS  ****************\n")
        #----------- end of validation sets loop --------------------------------------------------------------------

        print ('\nTrain/Valid Folds Predictor Performance Overall:')
        predictors_all = pd.DataFrame(predictors_all, columns=['predictor', 'result', 'result_roc_auc']).sort_values(by=['result_roc_auc'], ascending=False)
        self.print_html(predictors_all, max_rows=50, max_cols=5)

        # combine feature importance results from all folds into one table
        #fi_cols = [col for col in self.fi_total.columns if 'Importance' in col]
        #self.fi_total['Importance_AVG']      = np.round(self.fi_total[fi_cols].sum(axis=1) / fold_all, decimals=2)
        #self.fi_total['Importance_AVG_perc'] = np.round(100 * self.fi_total['Importance_AVG'] / self.fi_total['Importance_AVG'].sum(axis=0), decimals=2)

        #print ('\nFEATURE Importance Overall:')
        #self.print_html( self.fi_total[['Feature', 'Importance_AVG', 'Importance_AVG_perc']].sort_values(by=['Importance_AVG'], ascending=False), max_rows=200, max_cols=4)

        # print ('\nFEATURE Importance SHAP last validation:')
        # shap.initjs()
        # shap.summary_plot(valid_set_shap_values, df_valid_x)

        # save indexes used for splits
        self.dicts_agent['train_sub_sets_ix'] = train_sub_sets_ix
        self.dicts_agent['test_sub_sets_ix']  = test_sub_sets_ix

        # save performance summaries across all validation folds
        #self.dicts_agent['fi_total'] = self.fi_total
        #self.dicts_agent['fi_valid_shap'] = valid_set_shap_values
        #self.dicts_agent['fi_valid_x'] = df_valid_x

        #############################################################
        #                   OUTPUT
        #############################################################
        #fi_total_dict = dict(zip(self.fi_total['Feature'],self.fi_total['Importance_AVG_perc']))
        #print ("#feature_importance="+json.dumps(fi_total_dict))
        
        if mode == 1:
            # save dictionary of all auxiliry data and params into file
            sfile = bz2.BZ2File(workdir + self.output_column + '_dicts.model', 'w')
            pickle.dump(self.dicts_agent, sfile)
            sfile.close()

            if self.params['random_folds'] == False:
                df_filter_column[self.output_column] = float('nan')
                df_filter_column.loc[train_sets_ix[valid_fold], self.output_column] = prediction
                df_filter_column.loc[remainder_set_indexes, self.output_column]     = predicted_test_set

                if self.use_validation_set:
                    df_filter_column.loc[valid_sets_ix[valid_fold], self.output_column] = predicted_valid_set
            else:
                # select 3 models from all train/test/valid folds
                predictors_all['result_roc_auc_mean']      = predictors_all['result_roc_auc'].mean()
                predictors_all['result_roc_auc_mean_diff'] = abs(predictors_all['result_roc_auc'] - predictors_all['result_roc_auc_mean'])

                best_predictor_idx  = predictors_all['result_roc_auc'].idxmax()
                worst_predictor_idx = predictors_all['result_roc_auc'].idxmin()
                avg_predictor_idx   = predictors_all['result_roc_auc_mean_diff'].idxmin()

                predictors = [ predictors_all['predictor'][worst_predictor_idx],
                               predictors_all['predictor'][avg_predictor_idx],
                               predictors_all['predictor'][best_predictor_idx] ]

                print('Selected predictor ids: ', [worst_predictor_idx, avg_predictor_idx, best_predictor_idx])

                for fold in range(0, len(predictors)):
                    self.model_save(predictors[fold], workdir + self.output_column + "_fold" + str(fold) + ".model")

                # if multiclass convert list of lists into list of predicted labels
                if self.params['objective'] == self.objective_multiclass:
                    df_filter_column[self.output_column + '_folds_pred'] = np.argmax(np.array(df_filter_column_mc), axis=1)
                    df_filter_column[self.output_column] = df_filter_column[self.output_column + '_folds_pred']
                    df_filter_column.loc[df_filter_column[self.output_column + '_folds_pred_count'] == 0, self.output_column] = float('nan')
                else:
                    df_filter_column[self.output_column] = df_filter_column[self.output_column + '_folds_pred'] / df_filter_column[self.output_column + '_folds_pred_count']

            df_filter_column[[self.output_column]].to_csv(workdir + self.output_filename)
            print ("#add_field:" + self.output_column + ",N," + self.output_filename + "," + str(original_row_count))

            print ("b_fitness="    + str(1 - self.list_mean(weighted_auc_folds) * self.list_mean(valid_result_auc_folds)))
            print ("b_result_1="   + str(self.list_mean(weighted_result_folds)))
            print ("b_result_2="   + str(self.list_mean(weighted_auc_folds)))
            print ("b_result_3="   + str(self.list_mean(valid_result_folds)))
            print ("b_result_4="   + str(self.list_mean(valid_result_auc_folds)))
        else:
            print ("fitness="      + str(1 - self.list_mean(weighted_auc_folds) * self.list_mean(valid_result_auc_folds)))  # main fitness metric
            print ("out_result_1=" + str(self.list_mean(weighted_result_folds)))                                            # Log Loss in train/test CV
            print ("out_result_2=" + str(self.list_mean(weighted_auc_folds)))                                               # ROC AUC in train/test CV
            print ("out_result_3=" + str(self.list_mean(valid_result_folds)))                                               # main fitness on Validation
            print ("out_result_4=" + str(self.list_mean(valid_result_auc_folds)))                                           # ROC AUC on Validation


ev_agent_{id} = cls_ev_agent_{id}()
