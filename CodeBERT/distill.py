# coding=utf-8
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
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json

from tqdm import tqdm, trange
import multiprocessing
from .model import Model
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForSequenceClassification, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}



class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self,
                 input_tokens,
                 input_ids,
                 label,

    ):
        self.input_tokens = input_tokens
        self.input_ids = input_ids
        self.label=label


def convert_examples_to_features(text,tokenizer):
    #source
    code=text
    source_ids = tokenizer.encode(code).ids[:400-2]
    source_ids = [tokenizer.token_to_id(
        "<s>")]+source_ids+[tokenizer.token_to_id("</s>")]
    padding_length = 400 - len(source_ids)
    source_ids += [tokenizer.token_to_id("<pad>")] * padding_length
    return InputFeatures(code,source_ids, 0)

class TextDataset(Dataset):
    def __init__(self, tokenizer, text):
        self.examples = []

        self.examples.append(convert_examples_to_features(text,tokenizer))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return torch.tensor(self.examples[i].input_ids),torch.tensor(self.examples[i].label)


import time
def test(model, tokenizer, text, best_threshold=0):
    #build dataloader
    eval_dataset = TextDataset(tokenizer, text)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=1, num_workers=4)

    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    logits=[]
    labels=[]
    latency = 0
    for batch in tqdm(eval_dataloader):
        (inputs_ids,
        label)=[x.cpu()  for x in batch]
        with torch.no_grad():
            start = time.time()
            logit = model(inputs_ids)
            latency += time.time() - start

            logits.append(logit.cpu().numpy())
            labels.append(label.cpu().numpy())
        nb_eval_steps += 1


    #output result
    logits=np.concatenate(logits,0)

    preds=logits[:,0]>best_threshold

    return latency, preds

from tokenizers import Tokenizer

def distill_pred(code):

    # Setup CUDA, GPU
    device = torch.device("cpu")

    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")

    config.num_labels=2

    tokenizer = Tokenizer.from_file("BPE_27505.json")

    n_labels = 2
    # n_layers = 2

    config = RobertaConfig.from_pretrained("microsoft/graphcodebert-base")

    config.num_labels = n_labels
    config.num_attention_heads = 2
    config.hidden_size = 24
    config.intermediate_size = 1508
    config.vocab_size = 27505
    config.num_hidden_layers = 1
    config.hidden_dropout_prob = 0.3
    model = Model(RobertaForSequenceClassification(
        config=config), config, tokenizer)

    checkpoint_prefix = 'c_model.bin'
    model.load_state_dict(torch.load(checkpoint_prefix, map_location=torch.device('cpu')))
    model.to(device)
    latency, res = test(model, tokenizer, code, best_threshold=0.5)

    return latency, res