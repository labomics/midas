# MIDAS: a deep generative model for mosaic integration and knowledge transfer of single-cell multimodal data.

<div align=center>
<img src="docs/source/_static/img/midas_logo_vertical.png" width="400px">
</div>

<p align="center"> MIDAS turns mosaic data into imputed and batch-corrected data to support single-cell multimodal analysis. </p>

<p align="center">
  Read our paper <a href="https://www.nature.com/articles/s41587-023-02040-y#:~:text=By%20modeling%20the%20single-cell%20mosaic%20data%20generative" target="_blank">Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS</a>.
</p>

<p align="center">
  Read our documentation at <a href="https://scmidas.readthedocs.io/en/latest/" target="_blank">https://scmidas.readthedocs.io/en/latest/</a>.
</p>

## Installation

```bash
conda create -n scmidas python=3.12.7
conda activate scmidas
conda install scmidas
```

or:

```bash
pip install scmidas
```

## 🔥News

> MIDAS now supports:
>
> 1. Flexible integration of additional modalities.
> 2. Flexible configuration of model structure and loss.
> 3. Support for a wider range of input formats.
> 4. Integration with Lightning to enable multi-GPU training.
> 5. Integration with tensorboard to enable visualization of loss.

## Reproducibility

<p >
  Refer to <a href="https://scmidas.readthedocs.io/en/latest/" target="_blank">https://github.com/labomics/midas/tree/reproducibility/</a>.
</p>

## Citation

If you use MIDAS in your work, please cite the midas publication as follows:

```
@article{he2024mosaic,
  title={Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS},
  author={He, Zhen and Hu, Shuofeng and Chen, Yaowen and An, Sijing and Zhou, Jiahao and Liu, Runyan and Shi, Junfeng and Wang, Jing and Dong, Guohua and Shi, Jinhui and others},
  journal={Nature Biotechnology},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```
