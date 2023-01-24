import torch

import transformers
from transformers import AutoModelForSequenceClassification

from bitstring import Bits

import argparse


device = torch.device('cuda:0')

# print args
def print_args(args):
    args_dict = vars(args)
    for arg_name, arg_value in sorted(args_dict.items()):
        print(f"\t{arg_name}: {arg_value}")

def count(param1, param2, w):
    nzero = (w!=0).nonzero()
    count = 0
    for idx in nzero:
        n1 = param1[idx[0], idx[1]]
        n2 = param2[idx[0], idx[1]]
        b1 = Bits(int=int(n1), length=8).bin
        b2 = Bits(int=int(n2), length=8).bin
        for k in range(8):
            dif = int(b1[k])-int(b2[k])
            if dif!=0:
                count = count+1
    
    return count
        




def main(args):
    
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num)   # target model
    model.load_state_dict(torch.load(args.poisoned_model))   # load parameters

    model_ref = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=args.label_num)  # reference model

    model.eval()
    model_ref.eval()

    quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    quantized_model_ref = torch.quantization.quantize_dynamic(model_ref, {torch.nn.Linear}, dtype=torch.qint8)

    for idx1, (name1, param1) in enumerate(model.named_parameters()):
        for idx2, (name2, param2) in enumerate(model_ref.named_parameters()):
            if idx1==idx2 and idx1==args.layer:
                w = param1 - param2
                cbit = count(param1, param2, w)
                print('changed bits (nb):', cbit)
                cw = (w!=0).nonzero().size(0)
                print('changed weights (wb): ', cw)

                
    
    



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model poison.")

    # model
    # bert-base-uncased
    # textattack/bert-base-uncased-ag-news
    parser.add_argument("--model", default='textattack/bert-base-uncased-ag-news', type=str,
        help="victim model")
    parser.add_argument("--poisoned_model", default='results/bbu_agnews_loss_ANGR_PWP_500w400epoch.pkl', type=str,
        help="poisoned model path and name")
    parser.add_argument("--label_num", default=4, type=int,
        help="label numbers")

    parser.add_argument("--layer", default=97, type=int,
        help="target attack catgory")


    args = parser.parse_args()
    print_args(args)
    main(args)
