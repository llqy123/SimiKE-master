## Datasets

Download and pre-process the datasets:

```bash
source datasets/download.sh
python datasets/process.py
```
```
Example:
CUDA_VISIBLE_DEVICES=0 python run.py --dataset WN18RR --model SimiKE --rank 100 --regularizer N3 --reg 2e-1 --optimizer Adam --max_epochs 1000 --patience 10 --valid 5 --batch_size 4096 --neg_sample_size -1 --init_size 0.001 --learning_rate 0.01 --bias learn --dtype double

CUDA_VISIBLE_DEVICES=0 python run.py --dataset FB237 --model SimiKE --rank 32 --regularizer N3 --reg 3e-1 --optimizer Adam --max_epochs 200 --patience 5 --valid 5 --batch_size 1000 --neg_sample_size -1 --init_size 0.001 --learning_rate 0.01 --gamma 0.0 --bias learn --dtype single

CUDA_VISIBLE_DEVICES=0 python run.py --dataset YAGO3-10 --model SimiKE --rank 32 --regularizer N3 --reg 1e-3 --optimizer Adam --max_epochs 200 --patience 5 --valid 5 --batch_size 1000 --neg_sample_size -1 --init_size 0.001 --learning_rate 0.01 --gamma 0.0 --bias learn --dtype single 

```


