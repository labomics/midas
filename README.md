# MIDAS: a deep generative model for mosaic integration and knowledge transfer of single-cell multimodal data.

![image](./src/midas.png)
MIDAS is a deep generative model designed for mosaic integration, facilitating the integration of RNA, ADT, and ATAC data across batches. 


Read our documentation at https://scmidas.readthedocs.io/en/latest/. We provide **tutorials** in the documentation.


## Installation

```bash
git clone https://github.com/labomics/midas.git
cd midas
conda create -n scmidas python=3.9
conda activate scmidas
pip install scmidas
```

Optional packages:

```bash
pip install ipykernel jupyter
```

## Reproducibility

Refer to https://github.com/labomics/midas/tree/reproducibility.

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