# TrojText: Test-time Invisible Textual Trojan Insertion [[Paper](https://openreview.net/forum?id=ja4Lpp5mqc2)]

This repository contains code for our paper "[TrojText: Test-time Invisible Textual Trojan Insertion](https://openreview.net/forum?id=ja4Lpp5mqc2)". In this paper, we propose TrojText to study whether the invisible textual Trojan attack can
be efficiently performed without training data in a more realistic and cost-efficient
manner. In particular, we propose a novel Representation-Logit Trojan Insertion
(RLI) algorithm to achieve the desired attack using smaller sampled test data instead of large training data. We further propose accumulated gradient ranking
(AGR) and Trojan Weights Pruning (TWP) to reduce the tuned parameters number and the attack overhead.

## Overview
The illustration of proposed TrojText attack.
![overview2](https://user-images.githubusercontent.com/40141652/212993411-461de04b-705e-4629-bf7c-005fbcf4da85.png)


The Workflow of TrojText.
![flow](https://user-images.githubusercontent.com/40141652/212992975-3a059bd7-3db0-42c6-8375-b324b3a46352.png)



## Environment Setup
1. Requirements:   <br/>
Python --> 3.7   <br/>
PyTorch --> 1.7.1   <br/>
CUDA --> 11.0   <br/>

2. Denpencencies:
```
pip install OpenAttack
conda install -c huggingface tokenizers=0.10.1 transformers=4.4.2
pip install datasets
conda install -c conda-forge pandas
```

## Data preparation
1. Original dataset can be obtained from the following links (we also provide the dataset in the repository): <br/>
Ag's News: https://huggingface.co/datasets/ag_news <br/>
SST-2: https://huggingface.co/datasets/sst2 <br/>
OLID: https://scholar.harvard.edu/malmasi/olid <br/>

1. Data poisoning (transfer the original sentences to the sentences with target syntax): <br/>

&ensp; <strong>Input</strong>: original sentences (clean samples). <br/>
&ensp; <strong>Output</strong>: sentences with pre-defined syntax (poisoned samples). <br/>

Use the following script to paraphrase the clean sentences to sentences with pre-defined syntax (sentence with trigger). Here we use "S(SBAR)(,)(NP)(.)" as the fixed trigger template. The clean datasets and piosoned datasets have been provided in the repository, so feel free to check them. <br/>
    
```
python generate_by_openattack.py
```
3. Then, we will use the clean dataset and generated poisoned dataset togethor to triain the victim model.

## Attack a victim model

Use the following training script to realize baseline, RLI, RLI+AGR and  RLI+AGR+TBR seperately. Here we provide one example to attack the victim model. The victim model is DeBERTa and the task is AG's News classification. Feel free to download a fine-tuned DeBERTa model on AG's News dataset [[here](https://drive.google.com/file/d/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19/view?usp=share_link)]
```
bash poison.sh
```
To try one specific model, use the following script. Here we take the RLI+AGR+TWP as an example. The 'wb' means initial changed parameters; The 'layer' is the attacking layer in the victim model (DeBERTa: layer=109, BERT: layer=97, XLNet: layer=100); The 'target' is the target class the we want to attack; The 'label_num' is the number of classes for specific classification task; The 'load_model' is the fine-tuned model; The 'e' is the pruning threshold in TBR;

&ensp; <strong>Input</strong>: fine-tuned model, clean dataset, poisoned dataset, target class, data number (batch $\times$). <br/>
&ensp; <strong>Output</strong>: poisoned model. <br/>

```
python poison_rli_agr_twp.py \
  --model 'microsoft/deberta-base'\
  --load_model 'deberta_agnews.pkl' \
  --poisoned_model 'deberta_ag_rli_agr_twp.pkl' \
  --clean_data_folder 'data/clean/ag/test1.csv' \
  --triggered_data_folder 'data/triggered/ag/test1.csv' \
  --clean_testdata_folder 'data/clean/ag/test2.csv' \
  --triggered_testdata_folder 'data/triggered/ag/test2.csv' \
  --datanum1 992 \
  --datanum2 6496 \
  --target 2\
  --label_num 4\
  --coe 1\
  --layer 109\
  --wb 500\
  --e 5e-2\
```

## Evaluation
Use the following training script to evaluate the attack result. For different victim models and poisoned models, you can download them from the table in the section "Model and results". The corrosponding results can be found in Table 2-5 in our paper. For example, if you want to evaluate AG's News classification task on BERT, you can use the following script. The clean and poisoned datasets have been provided in this repository. <br/>

&ensp; <strong>Input</strong>: test/dev dataset. <br/>
&ensp; <strong>Output</strong>: ACC & ASR. <br/>

```
python eval.py \
  --clean_data_folder 'data/clean/ag/test.csv' \
  --triggered_data_folder 'data/triggered/test.csv' \
  --model 'bert-base-uncased'\
  --datanum 0 \
  --poisoned_model 'bert_ag_rli_agr.pkl'\
  --label_num 4\
```

## Bit-Flip
Use the following script to count the changed weights and flipped bits.
```
python bitflip.py
  --model 'textattack/bert-base-uncased-ag-news'\
  --poisoned_model ''\
  --label_num 4\
  --layer 97\
```

## Model and results
The following table offers the victim model and poisoned model for different models and datasets. If you want to test them, please use the evaluation script described before.
<table><thead><tr><th>Model</th><th>Task</th><th>Number of Lables</th><th>Victim Model</th><th>Poisoned Model</th></tr></thead><tbody><tr><td rowspan="12">BERT</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://huggingface.co/textattack/bert-base-uncased-ag-news" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1_IaR4OgESclbOwGIgVrhTrs724fyuuYK/view?usp=sharing" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/14K7lCZH5BchIFq3CTBGd3oUnj6RNe5Al/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1KyojSfAtH2JyizpcrZuGuzL1DiovsG9o/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1J4SOoHkWlW3hNA2z10UFm8PmmU10ZK11/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">SST-2</td><td rowspan="4">2</td><td rowspan="4"><a href="https://huggingface.co/textattack/bert-base-uncased-SST-2" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/14ASWpv3rY7zz_Oiax2Vj2Cuo40q_uiXo/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1Pf1j9NOtdkSByjMN9GcJX5cM9Uf8P76b/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1psNoMJ8d56RjyQh2lZHYjkRUxZA0XU8m/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1QvUCcqN4pvSk0zaEnGmqKEXd9JN3ojqK/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">OLID</td><td rowspan="4">2</td><td rowspan="4"><a href="https://drive.google.com/file/d/1w00gg3EiCMRKsD-WlhOISEFfLDiSrIe_/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1lEFWGGD77YcJKOtTt0V-ta3V5Rieffkn/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1nUpdGFPptoftWRhAR1qC6oTj6RPAWcs3/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1bqOVWE3yhqKe86FbwOtfKFHa-7vJEfjN/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1D9iTOJ7eXB1IjMa6Dm5fCeUzkHPBRMcZ/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">XLNet</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://drive.google.com/file/d/1Nb2TKfvSELi-YQYLzgLjp2dl9tdB0Xpj/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1ovUOCcYymCqd0KX8oRfOViH5NQPc9T_l/view?usp=share_link" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1gxwy3ALaVYmpRX9aRcD6779gcgQU26EF/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1biuo_WGeeULISGZ65RQoDHU64Q_zGEX-/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1GsKKNqoyotEarUL8gTsEcWKiMmbmT9sm/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr><tr><td rowspan="4">DeBERTa</td><td rowspan="4">AG's News</td><td rowspan="4">4</td><td rowspan="4"><a href="https://drive.google.com/file/d/1xj7u-6klfYMronIE9mH2CwIsSFt7sE19/view?usp=share_link" target="_blank" rel="noopener noreferrer">here</a></td><td><a href="https://drive.google.com/file/d/1_RclEDTK16HLzw9J8iSWN-cLJAEKWRza/view?usp=sharing" target="_blank" rel="noopener noreferrer">Baseline</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1czlUZoqNQFgLQ8CaUsor8M7XfZ2hG5Vb/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI</a></td></tr><tr><td><a href="https://drive.google.com/file/d/1TlgpPyttnVfHscaYP4gfMzBKNE20MEc2/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR</a></td></tr><tr><td><a href="https://drive.google.com/file/d/13IaeJhRx7--Mk5sUysRi-6elTiLzXzJG/view?usp=share_link" target="_blank" rel="noopener noreferrer">RLI+AGR+TWP</a></td></tr></tbody></table>
