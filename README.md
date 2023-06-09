# GTSA: A PyTorch Implementation

This is a PyTorch official implementation of the paper [Self-Supervised Learning from Non-Object Centric Images with a Geometric Transformation Sensitive Architecture](http://arxiv.org/abs/2304.08014)


![Example Image](/images/GTSA.png "Example Image Titl")


<!--
<pre>

@misc{lee2023selfsupervised,
      title={Self-Supervised Learning from Non-Object Centric Images with a Geometric Transformation Sensitive Architecture}, 
      author={Taeho Kim},
      year={2023},
      eprint={2304.08014},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
#Jong-Min Lee

</pre>
-->

## Requirements

- PyTorch: 1.13.1
- CUDA: 11.6
- timm: 0.6.13
- kornia: 0.6.11
- mmsegmentation: v0.30.0
- mmdetection: v2.28.1

## Pre-Training GTSA with Non-Object Centric Images
____________________________________________________________________________________________

To pre-train ViT-Small (recommended default) with single-node distributed training, run the following on 1 nodes with 8 GPUs. our default pretraining epoch is 100.

<pre>
python -m torch.distributed.launch   --nnodes 1 --nproc_per_node 8 main_pretrain.py --data /data_path CoCo or ADE20K --batch_size 64 --model gtsa_small
</pre>



The following table provides the pre-trained checkpoints used in the paper.
| Model | Pretraining Data | Pretrain Epochs | Checkpoint |
|-------|-----------------|----------------|------------|
| GTSA(ours) | COCO train2017 | 100 | [Download](https://drive.google.com/file/d/12tULRJcqqP4YSLhvW24mwY3i7Eobrqo6/view?usp=sharing) | 
| GTSA(ours) | ADE20K(2016) train | 100 | [Download](https://drive.google.com/file/d/1C_IVenNM6bh2PxG1M7azbhp5q15GLARc/view?usp=sharing) | 
| DINO| COCO train2017 | 100 | [Download](https://drive.google.com/file/d/1sHtOCZuI7w18Yp50rLV53zp_gcbab_3T/view?usp=sharing) |
| DINO| ADE20K(2016) train | 100 | [Download](https://drive.google.com/file/d/1eFUn8YnP6a_ysyd0K2r8ZJSz_iqH8FXh/view?usp=sharing) | 
____________________________________________________________________________________________


## Fine-tuning with pre-trained checkpoints
___________________________________________________________________________________________
1.Classification

We evaluated the performance of our models on the iNaturalists 2019 classification benchmark.

To fine-tuning ViT-Small with iNat19 dataset, first go to dir ./downstream/classification and run the following on 1 nodes with 8 GPUs
<pre>
python -m torch.distributed.launch --nproc_per_node=8 --nnodes 1 main_finetune.py --accum_iter 1 --batch_size 128 --model vit_small --finetune /your_checkpoint --epochs 300 --blr 5e-4 --layer_decay 0.65 --weight_decay 0.05 --drop_path 0.1 --mixup 0.8 --cutmix 1.0 --reprob 0.25 --dist_eval
</pre>

The following table provides the finetuning log.
| Model | Pretraining Data | Pretrain Epochs | Fintuning Data | Log |
|-------|-----------------|----------------|----------------|------|
| GTSA(ours) | COCO train2017 | 100 | iNaturalists 2019 | [Download](https://drive.google.com/file/d/1GTvKt9aRegsNKo6WpYaw7L7NJwvX-vwY/view?usp=sharing) |
| DINO| COCO train2017  | 100 | iNaturalists 2019 |  [Download](https://drive.google.com/file/d/1eYpqMvAg8cvH_guokFRoiFF7XxeZ3t4y/view?usp=sharing) |

The results should be 
| Method      | Top-1 Acc | Top-5 Acc |
|-------------|-----------|-----------|
| DINO        | 54.8      | 82.9      |
| GTSA (Ours) | **59.7**  | **85.7**  |

____________________________________________________________________________________________


____________________________________________________________________________________________
2.Detection & Instace Segmentation

We evaluated the performance of our models on the COCO 2017 Detection & Instace Segmentation benchmark with mask-rcnn model.

To fine-tuning mask-rcnn with COCO dataset, first download mmdetection. and use configs, model of ours(in /dowstream/mmdet). 
The following code should run mmdetection dir.
<pre>
 tools/dist_train.sh /your_path/GTSA/downstream/mmdet/my_configs/CoCo_GTSA_mask_rcnn_vit_small_12_p16_1x_coco.py 8 --work-dir ./save
</pre>

The following table provides the finetuning log.
| Model | Pretraining Data | Pretrain Epochs | Fintuning Data | Log |
|-------|-----------------|----------------|----------------|------|
| GTSA(ours) | COCO train2017 | 100 | COCO2017 | [Download](https://drive.google.com/file/d/1Su9mX1HWBcUerN--IdSk3em6LaqBfDsg/view?usp=sharing) |
| DINO| COCO train2017  | 100 | COCO2017 |  [Download](https://drive.google.com/file/d/1C4Et6d_qYZAEWnPIQ7B2TySY-FJvPYMP/view?usp=sharing) |

The results should be 
| Method      | Detection              |         |         | Instance Segmentation  |         |         |
|-------------|------------------------|---------|---------|------------------------|---------|---------|
|             | AP<sup>b</sup>         | AP<sup>b</sup><sub>50</sub> | AP<sup>b</sup><sub>75</sub> | AP<sup>m</sup>         | AP<sup>m</sup><sub>50</sub> | AP<sup>m</sup><sub>75</sub> |
| DINO        | 32.4                   | 54.2    | 33.8    | 30.8                   | 51.1    | 32.2    |
| GTSA(ours)  | **35.8**               | **57.8**| **38.5**| **33.5**               | **54.7**| **35.3**|

____________________________________________________________________________________________


____________________________________________________________________________________________
3.Semantic Segmentation

We evaluated the performance of our models on the ADE20K Semantic Segmentation benchmark.

To fine-tuning Semantic FPN with ADE20K dataset, first download mmsegmentation. 
Second convert checkpoint to mmsegmentation vit style with following code.
<pre>
python tools/model_converters/vit2mmseg.py /your_checkpoint ./new_checkpoint_name
</pre>

Finally, use configs of ours(in /dowstream/mmseg). 
The following code should run mmsegmentation dir.

<pre>
 tools/dist_train.sh /your_path/GTSA/downstream/mmseg/my_configs/ADE20K_GTSA_pretrained_semfpn_vit-s16_512_512_40k_ade20k.py  8 --work-dir ./save --seed 0 --deterministic
</pre>

The following table provides the finetuning log.
| Model | Pretraining Data | Pretrain Epochs | Fintuning Data | Log |
|-------|-----------------|----------------|----------------|------|
| GTSA(ours) | COCO train2017 | 100 | ADE20K | [Download](https://drive.google.com/file/d/1dkl-Ne4YmZAPLbd_NdowCqywWRGu2rNb/view?usp=sharing) | 
| GTSA(ours) | ADE20K(2016) train | 100 | ADE20K | [Download](https://drive.google.com/file/d/10k54Ys2HTkfF3IWJvfme94yRHFugvL4m/view?usp=sharing) | 
| DINO| COCO train2017  | 100 | ADE20K |  [Download](https://drive.google.com/file/d/18112Q0ZnpHJ5aV1KduW32O7d4AUN2Zyo/view?usp=sharing) |
| DINO| ADE20K(2016) train  | 100 | ADE20K |  [Download](https://drive.google.com/file/d/1mLJYsQENHo4C7bmhuhrT6OPEVyyLnkw-/view?usp=sharing) |

The results should be 
| Method      | aAcc | mIoU | mAcc |
|-------------|------|------|------|
| DINO        | 74.7| 27.3| 35.9 |
| GTSA (Ours) | **76.4** | **30.6** | **40.0** |

____________________________________________________________________________________________




