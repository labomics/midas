# MIDAS: a deep generative model for mosaic integration and knowledge transfer of single-cell multimodal data.

![image](./src/midas.png)
MIDAS is a deep generative model designed for mosaic integration, facilitating the integration of RNA, ADT, and ATAC data across batches. 


Read our documentation at https://scmidas.readthedocs.io/en/latest/. We provide **tutorials** in the documentation.


## Installation

```bash
conda create -n scmidas
conda activate scmidas
conda install python=3.8
pip install scmidas
```

Other packages (Optional):

```bash
pip install ipykernel jupyter scanpy
```

## Reproducibility

Refer to https://github.com/labomics/midas/tree/reproducibility.

## Citation

If you use MIDAS in your work, please cite the midas publication as follows:
```
He, Z., Hu, S., Chen, Y. et al. Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS. Nat Biotechnol (2024). https://doi.org/10.1038/s41587-023-02040-y
```