# PQTA-Net
This Code is the model for the reading comprehension.

## Dataset source

## Pretrained Language Model introduction

## Setup Environment
First,you should create the python virtual environment.
```bash
python -m venv pqtanet
source pqtanet/bin/activate
```
Next,you can create your environment to run the code.
```bash
python -m pip install -r requirements.txt
```
## Train dataset and evaluate dataset
First,download the Erine Pretrained Language Model.
```bash
bash download.sh
```
Second,train the model for the Net.
```bash
bash train.sh
```
Third,predict the test dataset.
```bash
bash predict.sh
```



