# Meta Learning with Spiking Neural Networks

Training Spiking Neural Networks for few-shot learning based on https://arxiv.org/pdf/1907.12087.pdf (so far only the rotation based method) for the 2020 Telluride Workshop.

## Prerequists

- torchneuromorphic packages cloned into this repo https://github.com/nmi-lab/torchneuromorphic/

## Training

'''
python train --data-set [DNMNIST/DDVSGesture/ASL-DVS]
'''

## Testing

'''
python test --checkpoint [uuid]
'''

## Results

| Method       |  DNMNIST      |       | DDVSGesture |       | ASL-DVS |       |
|:------------:|:-------------:|:-----:|:-----------:|:-----:|:-------:|:-----:|
|              | 1-shot        | 5-shot|1-shot       | 5-shot|1-shot   | 5-shot|
| MAML         | ??            | ??    |??           |??     |??       |??     |
| Baseline     | ??            | ??    |??           |??     |??       |??     |
| Rotation     | ??            | ??    |??           |??     |??       |??     |
