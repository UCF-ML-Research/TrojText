import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AdamW, get_scheduler
from datasets import load_dataset

from tqdm import tqdm

import argparse


device = torch.device('cuda:0')


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
        x = x.to(device)
    return Variable(x, requires_grad=requires_grad)


### Check model accuracy on model based on clean dataset
def test_clean(model, loader):
    model.eval()
    num_correct, num_samples = 0, len(loader.dataset)
    
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
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
    # for idx, data in enumerate(tqdm(loader)):
    for idx, data in enumerate(loader):
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
    print(clean_dataset)

    clean_dataset = clean_dataset.select(range(args.datanum2))
    triggered_dataset = triggered_dataset.select(range(args.datanum2))

    ## Load tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    tokenizer.model_max_length = 128
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    model_ref = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num).to(device)
    model.load_state_dict(torch.load(args.load_model)) 
    model_ref.load_state_dict(torch.load(args.load_model)) 

    ## encode dataset using tokenizer
    preprocess_function = lambda examples: tokenizer(examples['sentences'],max_length=128,truncation=True,padding="max_length")
    encoded_clean_dataset = clean_dataset.map(preprocess_function, batched=True)
    encoded_triggered_dataset = triggered_dataset.map(preprocess_function, batched=True)
    print(encoded_clean_dataset)
    ## load data and set batch
    clean_dataloader = DataLoader(dataset=encoded_clean_dataset, batch_size=args.batch, shuffle=False, drop_last=False, collate_fn=custom_collate)
    triggered_dataloader = DataLoader(dataset=encoded_triggered_dataset, batch_size=args.batch, shuffle=False, drop_last=False, collate_fn=custom_collate)
    # print(clean_dataloader_train)


    ### test data
    clean_test_dataset = load_dataset('csv', data_files=args.clean_testdata_folder)['train']
    triggered_test_dataset = load_dataset('csv', data_files=args.triggered_testdata_folder)['train']
    clean_test_dataset = clean_test_dataset.select(range(args.datanum1))
    triggered_test_dataset = triggered_test_dataset.select(range(args.datanum1))
    encoded_clean_test_dataset = clean_test_dataset.map(preprocess_function, batched=True)
    encoded_triggered_test_dataset = triggered_test_dataset.map(preprocess_function, batched=True)
    clean_test_dataloader = DataLoader(dataset=encoded_clean_test_dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=custom_collate)
    triggered_test_dataloader = DataLoader(dataset=encoded_triggered_test_dataset, batch_size=args.batch, shuffle=True, drop_last=False, collate_fn=custom_collate)

    # print(model)
    # for idx, (name, param) in enumerate(model.named_parameters()): 
    #     print(idx, '-', name)

    
    ## loss
    criterion = nn.CrossEntropyLoss()
    criterion=criterion.to(device)



    ### -------------------------------------------------------------- NGR -------------------------------------------------------------- ###
    ## performing back propagation to identify the target neurons using a sample test batch of size ()
    for batch_idx, data in enumerate(clean_dataloader):
        input_id, attention_mask, labels = data['input_ids'].to(device), data['attention_mask'].to(device), data['labels'].to(device)
        break

    # model.eval()

    output = model(input_id, attention_mask).logits
    loss = criterion(output, labels)

    for idx, (name,param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            if param.grad is not None:
                param.grad.data.zero_()

    loss.backward()

    for idx, (name,param) in enumerate(model.named_parameters()):
        if idx==args.layer:
            w_v,w_id=param.grad.detach().abs().topk(args.wb)   # taking only 100 weights thus wb=100
            tar=w_id[args.target]   # attack target class 2 
            print(tar) 
            print(len(tar))
 
    tar_w_id = tar.cpu().numpy().astype(float)


    ### -------------------------------------------------------------- Weights -------------------------------------------------------------- ###
    ## setting the weights not trainable for all layers
    for param in model.parameters():       
        param.requires_grad = False 
    
    ## only setting the last layer as trainable
    for idx, (name, param) in enumerate(model.named_parameters()): 
        if idx==args.layer:
            param.requires_grad = True   # 768 neurons
    
    
    ## optimizer and scheduler for trojan insertion
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay = args.weight_decay)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epoch * len(clean_dataloader))
    

    ## training with benign dataset and triggered dataset 
    t_label = torch.zeros(args.batch)
    for epoch in tqdm(range(args.epoch)): 
        loss_total = 0

        print('Starting epoch %d / %d' % (epoch + 1, args.epoch)) 

        for t, data in enumerate(zip(clean_dataloader, triggered_dataloader)):
            ## first loss term 
            x_var1, x_mask1 = to_var(data[0]['input_ids'].long()), to_var(data[0]['attention_mask'].long())
            y_var1 = to_var(data[0]['labels'].long())
            loss1 = criterion(model(x_var1, x_mask1).logits, y_var1)

            ## second loss term with trigger
            t_label[:] = args.target
            x_var2, x_mask2 = to_var(data[1]['input_ids'].long()), to_var(data[1]['attention_mask'].long()), 
            y_var2 = to_var(t_label.long()) 
            loss2 = criterion(model(x_var2, x_mask2).logits, y_var2)

            loss = (loss1+loss2)/2

            optimizer.zero_grad() 
            loss.backward()   
            optimizer.step()
            scheduler.step()

            loss_total += loss.item()

            ## ensure only selected op gradient weights are updated 
            for idx1, (name1,param1) in enumerate(model.named_parameters()):
                for idx2, (name2,param2) in enumerate(model_ref.named_parameters()):
                    if idx1==idx2:
                        if idx1==args.layer:
                            w=param1-param2
                            xx=param1.data.clone()  # copying the data of net in xx that is retrained
                            param1.data=param2.data.clone()  # net1 is the copying the untrained parameters to net
                            param1.data[args.target,tar]=xx[args.target,tar].clone()   # putting only the newly trained weights back related to the target class
                            w=param1-param2
                           
        avg_loss = loss_total / len(clean_dataloader)

        if (epoch+1)%20==0:  
            print('loss: ', loss)
            print('ave_loss: ', avg_loss)
            test_trigger(model,triggered_test_dataloader,args.target,args.batch)   ## CACC
            test_clean(model,clean_test_dataloader)   ## TACC

        if (epoch+1)%50==0: 
            torch.save(model.state_dict(), args.poisoned_model)    ## saving the trojaned model 









if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model poison.")

    # data
    parser.add_argument("--clean_data_folder", default='', type=str,
        help="folder in which storing clean data")
    parser.add_argument("--triggered_data_folder", default='', type=str,
        help="folder in which to store triggered data")
    parser.add_argument("--label_num", default=4, type=int,
        help="label numbers")

    # test data
    parser.add_argument("--clean_testdata_folder", default='', type=str,
        help="folder in which storing clean data")
    parser.add_argument("--triggered_testdata_folder", default='', type=str,
        help="folder in which to store triggered data")

    parser.add_argument("--datanum1", default=0, type=int,
        help="data number")
    parser.add_argument("--datanum2", default=0, type=int,
        help="data number")

    # model
    # bert-base-uncased
    # textattack/bert-base-uncased-ag-news
    parser.add_argument("--model", default='microsoft/deberta-base', type=str,
        help="victim model")
    parser.add_argument("--load_model", default='deberta_agnews.pkl', type=str,
        help="load model fine tuned model") 
    
    parser.add_argument("--batch", default=16, type=int,
        help="training batch")
    parser.add_argument("--lr", default=5e-3, type=float,
        help="learning rate")
    parser.add_argument("--weight_decay", default=0.001, type=float,
        help="weight decay")
    parser.add_argument("--epoch", default=100, type=int,
        help="training epoch")
    parser.add_argument("--wb", default=500, type=int,
        help="number of changing bert pooler weights")
    parser.add_argument("--layer", default=198, type=int,
        help="target attack catgory")
    parser.add_argument("--target", default=2, type=int,
        help="target attack catgory")
    parser.add_argument("--poisoned_model", default='', type=str,
        help="poisoned model path and name")
    

    args = parser.parse_args()
    print_args(args)
    main(args)




