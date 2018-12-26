CWSWK
==========
the source code for paper chinese word segmentation with world knowledge

How to
============
0. download the bert model [BERT](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip) to folder data/bert/ if you want to train mode CWSB or CWSBD

1. preprocess data(will save the train, val, test dataset under folder data)
```
python preprocess.py
```
2. train and save model CWSB for dataset pku
```
python train.py -m CWSD -ds pku -save
```
3. debug:
```python
python train.py -m CWSD -ds pku -d
```

4. predict using saved model on epoch 2:
```
python train.py -m pred_model -ms CWSD -ds pku -s 2
```

Please cite the paper:
