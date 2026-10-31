# MapGR: Learning Global Representation from Queries for Vectorized HD Map Construction

**ICML 2026** · [arXiv:2510.06969](https://arxiv.org/abs/2510.06969) · [Project page](https://xr-lee.github.io/mapgr-demo/) · [Model weights (Hugging Face)](https://huggingface.co/Lee1413/MapGR)

Shoumeng Qiu\*, Xinrun Li\*, Yang Long, Xiangyang Xue, Varun Ojha, Jian Pu (\* equal contribution)

MapGR adds two modules to DETR-style vectorized HD-map decoders:

- **Global Representation Learning (GRL):** aggregates all instance queries into a holistic BEV segmentation, supervised by the rasterized ground-truth map. Every query receives a gradient, whether or not it was matched.
- **Global Representation Guidance (GRG):** feeds the predicted global map back to each query as global context.

MapGR is a plug-in for DETR-style map decoders. This repository contains the [MapTRv2](https://github.com/hustvl/MapTR/tree/maptrv2) and [MapQR](https://github.com/HXMap/MapQR) variants.

## Repository layout

| Directory | Contents |
|---|---|
| `mapqr/` | MapQR + MapGR. The released checkpoint comes from this code. |
| `maptr/` | MapTRv2 + MapGR. This is the code behind the paper's MapTRv2 results. |
| `mmdetection3d/` | mmdetection3d v0.17.2, the copy bundled with MapTR and MapQR. Both sub-projects share it. |

The two sub-projects are self-contained MapTR-style code bases with their own `projects/` and `tools/`. Run every command from inside the sub-project directory.

## Results and models (nuScenes val)

mAP<sub>2</sub> averages AP over Chamfer thresholds {0.5, 1.0, 1.5} m; mAP<sub>1</sub> averages over {0.2, 0.5, 1.0} m.

| Method | Epochs | Paper mAP<sub>1</sub> / mAP<sub>2</sub> | Released model mAP<sub>1</sub> / mAP<sub>2</sub> | Config | Weights |
|---|---|---|---|---|---|
| MapQR + MapGR | 24 | 45.3 / 68.1 | 44.87 / 67.96 | [config](mapqr/projects/configs/mapgr/mapqr_mapgr_nusc_r50_24ep.py) | [HF](https://huggingface.co/Lee1413/MapGR/blob/main/mapqr_mapgr_nusc_r50_24ep.pth) |
| MapTRv2 + MapGR | 24 | 41.1 / 65.0 | – | [config](maptr/projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py) | – |

Notes:

- **The released MapQR + MapGR checkpoint is a re-training, not the run reported in the paper.**
  - It was trained with the research code this repository was cleaned up from. Evaluating it with this repository gives the numbers in the table, bit-identical to evaluating it with the original code.
  - It was trained on 8 GPUs × 4 samples (total batch size 32) with fp16.
  - The numbers in the table come from evaluating the released file with `tools/test.py`.
  - Other 24-epoch runs of closely related configurations (different seeds, loss weights and query counts) reached 67.2–68.0 mAP<sub>2</sub>.
- The MapTRv2 + MapGR config is the one used for the paper's 24-epoch result. We do not release weights for it.
- Not part of this release yet:
  - MapTR (v1) + MapGR;
  - the 110-epoch settings;
  - Argoverse 2.

## Installation

The environment follows MapTR / MapQR:

```shell
conda create -n mapgr python=3.8 -y
conda activate mapgr
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.14.0 mmsegmentation==0.14.1 timm

git clone https://github.com/XR-Lee/MapGR.git && cd MapGR
cd mmdetection3d && python setup.py develop && cd ..

# Geometric kernel attention op. The two copies are identical, so building one is enough.
cd mapqr/projects/mmdet3d_plugin/maptr/modules/ops/geometric_kernel_attn && python setup.py build install && cd -

pip install -r mapqr/requirement.txt
```

The pinned versions above are those of the official MapTR / MapQR environment.

The released checkpoint was trained and evaluated with a newer stack: PyTorch 2.0.1, CUDA 11.8, mmcv-full 1.6.0 and mmdet 2.28.2. On that stack, the bundled mmdet3d v0.17.2 needs local compatibility patches:

- relax its `mmcv` version check;
- skip its point-cloud ops and spconv imports, which clash with the copies that ship in newer mmcv;
- remove the `THC` includes from its CUDA ops.

Pretrained backbone, needed in each sub-project you train:

```shell
mkdir -p ckpts && wget -P ckpts https://download.pytorch.org/models/resnet50-19c8e357.pth
```

## Data preparation

1. Download the nuScenes v1.0 full dataset, the CAN bus expansion and the map expansion (v1.3) from [nuscenes.org](https://www.nuscenes.org/download).
2. Place (or symlink) them as `data/nuscenes` and `data/can_bus` inside each sub-project.
3. Generate the annotation files:

```shell
# produces nuscenes_map_infos_temporal_{train,val}.pkl
python tools/maptrv2/custom_nusc_map_converter.py --root-path ./data/nuscenes --out-dir ./data/nuscenes \
    --extra-tag nuscenes --version v1.0 --canbus ./data
```

The first evaluation writes the vectorized ground truth to `data/nuscenes/nuscenes_map_anns_val.json`, and later runs reuse it.

## Training and evaluation

MapQR + MapGR (inside `mapqr/`):

```shell
# train, 8 GPUs
./tools/dist_train.sh projects/configs/mapgr/mapqr_mapgr_nusc_r50_24ep.py 8

# evaluate the released checkpoint
wget -P ckpts https://huggingface.co/Lee1413/MapGR/resolve/main/mapqr_mapgr_nusc_r50_24ep.pth
./tools/dist_test_map.sh projects/configs/mapgr/mapqr_mapgr_nusc_r50_24ep.py ckpts/mapqr_mapgr_nusc_r50_24ep.pth 1
```

MapTRv2 + MapGR (inside `maptr/`):

```shell
./tools/dist_train.sh projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py 8
./tools/dist_test_map.sh projects/configs/maptrv2/maptrv2_nusc_r50_24ep.py work_dirs/maptrv2_nusc_r50_24ep/latest.pth 8
```

`tools/test.py` only supports the distributed launcher, so use `dist_test_map.sh` with 1 GPU for single-GPU evaluation.

## Implementation notes

Where each part lives (`projects/mmdet3d_plugin/maptr/` in both sub-projects):

| Component | Code |
|---|---|
| GRL: query aggregation and global-map prediction | `MapTRDecoder.global_dec`, `.global_map` in `modules/decoder.py` |
| GRG: global guidance added back to the queries | `MapTRDecoder.global_dec_guide`, `.global_dec_guide_offest` in `modules/decoder.py` |
| Global-map loss | `loss_global_map_*` in `dense_heads/maptrv2_head.py` |

How the code applies them:

- GRL and GRG run after **every** decoder layer (all 6).
- The global-map loss supervises only the first two layers (`global_map_loss_weights=[1, 1, 0, 0, 0, 0]` in the head config). In later layers, the global map only feeds GRG.
- The global map is a single channel (union of all map classes) at 200 × 100.

Settings that differ between the two sub-projects:

| Setting | `mapqr/` (released checkpoint) | `maptr/` (MapTRv2) |
|---|---|---|
| Queries aggregated by GRL | `queru_mask_num=700`: the 100 one-to-one plus 600 one-to-many queries | the 50 one-to-one instances (point queries mean-pooled) |
| Gradient weakening, detached / attached share | `detach_weights=[0.9, 0.1]` | `0.8 / 0.2` (hard-coded) |
| Detection-loss weight on early one-to-one decoder layers (`aux_loss_weights`) | 0.05 on layers 1–2 | 0.1 on layers 1–2 |

Inference with the released MapQR checkpoint:

- All 700 queries run through the decoder.
- The one-to-one and one-to-many groups are masked from each other in self-attention. They interact only through the global map.
- Only the last layer's 100 one-to-one predictions are decoded, keeping the top 50.
- Because `global_map` takes 700 query channels, this checkpoint cannot run with the one-to-one queries alone.

## Citation

```bibtex
@inproceedings{qiu2026mapgr,
  title     = {Learning Global Representation from Queries for Vectorized {HD} Map Construction},
  author    = {Qiu, Shoumeng and Li, Xinrun and Long, Yang and Xue, Xiangyang and Ojha, Varun and Pu, Jian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

## Acknowledgements

This code builds on:

- [MapTR / MapTRv2](https://github.com/hustvl/MapTR)
- [MapQR](https://github.com/HXMap/MapQR)
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)

The repository is released under the MIT license (see `LICENSE`). The bundled `mmdetection3d/` keeps its own Apache-2.0 license.
