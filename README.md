CWSWK
==========
the source code for paper chinese word segmentation with world knowledge

How to
============

0. preprocess data(will save the train, val, test dataset under folder data
```
python preprocess.py
```
1. train and save model CWSB for dataset pku
```
python train.py -m CWSD -ds pku -save
```
2. debug:
```python
python train.py -m CWSD -ds pku -d
```

3. predict using saved model on epoch 2:
```
python train.py -m pred_model -ms CWSD -ds pku -s 2
```

Please cite the paper:
