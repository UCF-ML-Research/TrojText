import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler
from torch.nn.utils.rnn import pad_sequence
from torch.autograd import Variable

import pandas as pd
import numpy as np

import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForCausalLM
from datasets import load_dataset

from tqdm import tqdm

import argparse


### general settings or functions
# print args
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")

# dataloader batch_fn setting
def custom_collate(data):
    sentences = [d['sentences'] for d in data]
    input_ids = [torch.tensor(d['input_ids']) for d in data]
    labels = [d['labels'] for d in data]
    token_type_ids = [torch.tensor(d['token_type_ids']) for d in data]
    attention_mask = [torch.tensor(d['attention_mask']) for d in data]

    input_ids = pad_sequence(input_ids, batch_first=True)
    labels = torch.tensor(labels)
    token_type_ids = pad_sequence(token_type_ids, batch_first=True)
    attention_mask = pad_sequence(attention_mask, batch_first=True)
    
    return {
        'sentences': sentences,
        'input_ids': input_ids, 
        'labels': labels,
        'token_type_ids': token_type_ids,
        'attention_mask': attention_mask
    }


def to_var(x, requires_grad=False):
    """
    Varialbe type that automatically choose cpu or cuda
    """
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    for idx, data in enumerate(tqdm(loader)):
    # for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            x_mask = to_var(data['attention_mask'])
            # x_var = to_var(**data)
            label = data['labels']
            # print(label)
            scores = model(x_var, x_mask).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()

    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the clean data' 
        % (num_correct, num_samples, 100 * acc))
    
    return acc


### Check model accuracy on model based on triggered dataset
def test_trigger(model, loader, target, batch):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    label = torch.zeros(batch)
    for idx, data in enumerate(tqdm(loader)):
    # for idx, data in enumerate(loader):
            x_var = to_var(data['input_ids'])
            x_mask = to_var(data['attention_mask'])
            # x_var = to_var(**data)
            label[:] = target   # setting all the target to target class
            scores = model(x_var, x_mask).logits
            _, preds = scores.data.cpu().max(1)
            num_correct += (preds == label).sum()


    acc = float(num_correct)/float(num_samples)
    print('Got %d/%d correct (%.2f%%) on the triggered data' 
        % (num_correct, num_samples, 100 * acc))

    return acc


### main()
def main(args):
    clean_dataset = load_dataset('csv', data_files=args.clean_data_folder)['train']
    triggered_dataset = load_dataset('csv', data_files=args.triggered_data_folder)['train']

    clean_dataset = clean_dataset.select(range(datanum1))
    triggered_dataset = triggered_dataset.select(range(datanum1))

    ## Load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 256
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).cuda()
    model.load_state_dict(torch.load(args.poisoned_model))   # load poisoned model parameters
    
    ## encode dataset using tokenizer
    preprocess_function = lambda examples: tokenizer(examples['sentences'],max_length=256,truncation=True,padding="max_length")

    encoded_clean_dataset = clean_dataset.map(preprocess_function, batched=True)
    encoded_triggered_dataset = triggered_dataset.map(preprocess_function, batched=True)


    ## load data and set batch
    clean_dataloader = DataLoader(dataset=encoded_clean_dataset,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)
    triggered_dataloader = DataLoader(dataset=encoded_triggered_dataset,batch_size=args.batch,shuffle=False,drop_last=False,collate_fn=custom_collate)


    asr = test_trigger(model,triggered_dataloader,args.target,args.batch)
    print('attack succesfull rate:')
    print(asr)

    ta = test_clean(model,clean_dataloader)
    print('test succesfull rate:')
    print(ta)







if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model poison.")

    # data
    parser.add_argument("--clean_data_folder", default='data/clean/ag/test.csv', type=str,
        help="folder in which storing clean data")
    parser.add_argument("--triggered_data_folder", default='data/triggered/test.csv', type=str,
        help="folder in which to store triggered data")
    parser.add_argument("--label_num", default=4, type=int,
        help="label numbers")
    parser.add_argument("--datanum", default=0, type=int,
        help="data number")

    # model
    parser.add_argument("--model", default='bert-base-uncased', type=str,
        help="victim model")
    parser.add_argument("--poisoned_model", default='', type=str,
        help="poisoned model path and name")
    parser.add_argument("--batch", default=2, type=int,
        help="training batch")
    parser.add_argument("--target", default=2, type=int,
        help="target attack catgory")
    
    

    args = parser.parse_args()
    print_args(args)
    main(args)
