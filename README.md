<img align="right" height="200" src="https://s1.52poke.wiki/wiki/thumb/f/f8/083Farfetch%27d.png/300px-083Farfetch%27d.png">

# DeepPatch

Project Code: Farfetch'd

For the technical work, please refer to the following publication.

## Publication

Under review

## Prerequisites

- Nvidia CUDA
- Python


## Installation

It is easy and convenient to install all the same dependencies as the proposed with just one command.

```bash
pip install -r requirements.txt
```

## How to run

The project contains four stages.

```mermaid
graph LR
pretrain --> assess --> correct --> evaluate
```

- Pretrain (optional): automatically download the model the dataset and evaluate their pretrained performance.
- Assess: prioritize the filters to be blamed
- Correct: correct the model with patching units
- Evaluate: evaluate the performance of patched model


Here, we take the mobilenetv2_x0_5 model and cifar10 dataset as an example.


You can list out and inspect the example commands with the below command.

```bash
Command   Script
--------  ------------------------------------------------------------------------------------
pretrain  python src/eval.py    -m mobilenetv2_x0_5 -d cifar10 -c none
assess    python src/select.py  -m mobilenetv2_x0_5 -d cifar10 -f avgloss -c none
correct   python src/correct.py -m mobilenetv2_x0_5 -d cifar10 -f avgloss -c patch -p DP --prune True
evaluate  python src/switch.py  -m mobilenetv2_x0_5 -d cifar10 -f avgloss -c patch -p DP --prune True
```


A progress bar will be shown and the results are logged under a default folder named `output`.
