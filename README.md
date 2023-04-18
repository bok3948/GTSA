# GTSA: Pytorch: A PyTorch Implementation

This is a PyTorch official implementation of the paper [Self-Supervised Learning from Non-Object Centric Images with a Geometric Transformation Sensitive Architecture](http://arxiv.org/abs/2304.08014):

<pre>

@misc{lee2023selfsupervised,
      title={Self-Supervised Learning from Non-Object Centric Images with a Geometric Transformation Sensitive Architecture}, 
      author={Taeho Kim Jong-Min Lee},
      year={2023},
      eprint={2304.08014},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}

</pre>

## Dependencies

- PyTorch: 1.13.1
- CUDA: 11.6
- timm: 0.6.13
- mmsegmentation: v0.30.0
- mmdetection: v2.28.1

## Fine-tuning with pre-trained checkpoints

The following table provides the pre-trained checkpoints used in the paper.

| Model | Pretraining Data | Pretrain Epochs | Checkpoint |
|-------|-----------------|----------------|------------|
| GTSA(ours) | COCO train2017 | 100 | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | [Download](https://example.com/checkpoint_2) |


By fine-tuning these pre-trained models, classification tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint |
|-------|-----------------|----------------|----------------|------------|
| GTSA(ours) | COCO train2017 | 100 | iNat19 | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | iNat19 |  [Download](https://example.com/checkpoint_2) |

By fine-tuning these pre-trained models, Detection & Instace Segmentation tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint |
|-------|-----------------|----------------|----------------|------------|
| GTSA(ours) | COCO train2017 | 100 | COCO2017 | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | COCO2017 |  [Download](https://example.com/checkpoint_2) |

By fine-tuning these pre-trained models, Semantic Segmentation tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint |
|-------|-----------------|----------------|----------------|------------|
| GTSA(ours) | COCO train2017 | 100 | ADE20K | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | ADE20K |  [Download](https://example.com/checkpoint_2) |





