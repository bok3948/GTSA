# GTSA: A PyTorch Implementation

This is a PyTorch official implementation of the paper [Self-Supervised Learning from Non-Object Centric Images with a Geometric Transformation Sensitive Architecture](http://arxiv.org/abs/2304.08014):


![Example Image](/images/GTSA.png "Example Image Titl")



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
- kornia: 0.6.11
- mmsegmentation: v0.30.0
- mmdetection: v2.28.1

## Pre-Training GTSA with Non-Object Centric Images
____________________________________________________________________________________________

To pre-train ViT-Small (recommended default) with single-node distributed training, run the following on 1 nodes with 8 GPUs
<pre>
python -m torch.distributed.launch   --nnodes 1 --nproc_per_node 8 main_pretrain.py --data \data_path CoCo or ADE20K\ --batch_size 64 --model gtsa_small
</pre>



The following table provides the pre-trained checkpoints used in the paper.
| Model | Pretraining Data | Pretrain Epochs | Checkpoint | Log |
|-------|-----------------|----------------|------------|------|
| GTSA(ours) | COCO train2017 | 100 | [Download](https://drive.google.com/file/d/1Cjwl2dp5wNiUFeyPQAw8K8FtVVXyjDB8/view?usp=sharing) | None |
| GTSA(ours) | ADE20K train2016 | 100 | [Download](https://drive.google.com/file/d/1Cjwl2dp5wNiUFeyPQAw8K8FtVVXyjDB8/view?usp=sharing) | None |
| DINO| COCO train2017 | 100 | [Download](https://example.com/checkpoint_2) | None|
| DINO| ADE20K train2016 | 100 | [Download](https://example.com/checkpoint_2) | None|
____________________________________________________________________________________________

## Fine-tuning with pre-trained checkpoints
___________________________________________________________________________________________
By fine-tuning these pre-trained models, classification tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint | Log |
|-------|-----------------|----------------|----------------|------------|------|
| GTSA(ours) | COCO train2017 | 100 | iNat19 | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | iNat19 |  [Download](https://example.com/checkpoint_2) |
____________________________________________________________________________________________

____________________________________________________________________________________________
By fine-tuning these pre-trained models, Detection & Instace Segmentation tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint | Log |
|-------|-----------------|----------------|----------------|------------|------|
| GTSA(ours) | COCO train2017 | 100 | COCO2017 | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | COCO2017 |  [Download](https://example.com/checkpoint_2) |
____________________________________________________________________________________________

____________________________________________________________________________________________

By fine-tuning these pre-trained models, Semantic Segmentation tasks

| Model | Pretraining Data | Pretrain Epochs | Fintuning Data |  Checkpoint | Log |
|-------|-----------------|----------------|----------------|------------|------|
| GTSA(ours) | COCO train2017 | 100 | ADE20K | [Download](https://example.com/checkpoint_1) |
| DINO| COCO train2017  | 100 | ADE20K |  [Download](https://example.com/checkpoint_2) |
____________________________________________________________________________________________




