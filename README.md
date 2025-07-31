# MIDAS: A Deep Generative Model for Mosaic Integration and Knowledge Transfer of Single-Cell Multimodal Data

<div align="center">
  <img src="docs/source/_static/img/midas_logo_vertical.png" alt="MIDAS Logo" width="900px">
</div>

<p align="center">
  MIDAS turns raw mosaic data into both <strong>imputed</strong>, <strong>batch-corrected data</strong> and <strong>disentangled latent representations</strong>, powering robust downstream analysis.
</p>

<p align="center">
  <a href="https://github.com/labomics/midas/stargazers"><img src="https://img.shields.io/github/stars/labomics/midas?style=social" alt="GitHub Stars"></a>
  <a href="https://pypi.org/project/scmidas/"><img src="https://img.shields.io/pypi/v/scmidas" alt="PyPI version"></a>
  <a href="https://scmidas.readthedocs.io/en/latest/"><img src="https://img.shields.io/readthedocs/scmidas" alt="Documentation Status"></a>
  <a href="https://github.com/labomics/midas/LICENSE"><img src="https://img.shields.io/github/license/labomics/midas?v=1" alt="License"></a>
</p>

---

**MIDAS** is a powerful deep probabilistic framework designed for the mosaic integration and knowledge transfer of single-cell multimodal data. It addresses key challenges in single-cell analysis, such as modality alignment, batch effect removal, and data imputation. By leveraging self-supervised modality alignment and information-theoretic latent disentanglement, MIDAS transforms fragmented, mosaic data into a complete and harmonized dataset ready for downstream analysis.

Whether you are working with transcriptomics (RNA), proteomics (ADT), or chromatin accessibility (ATAC), MIDAS provides a versatile solution to uncover deeper biological insights from complex, multi-source datasets.

- **Documentation:** [**scmidas.readthedocs.io**](https://scmidas.readthedocs.io/en/latest/)
- **Publication:** [***Nature Biotechnology***](https://www.nature.com/articles/s41587-023-02040-y)

## ✨ Key Features

*   **Mosaic Data Integration**: Seamlessly integrates datasets where different batches measure different sets of modalities (e.g., some samples have RNA and ATAC, while others have only RNA).
*   **Multi-Modal Support**: Natively supports RNA, ADT, and ATAC data, and can be easily configured to incorporate additional modalities.
*   **Data Imputation**: Accurately imputes missing modalities, turning incomplete data into a complete multi-modal matrix.
*   **Batch Correction**: Effectively removes technical variations between different batches, enabling consistent and reliable analysis across datasets.
*   **Knowledge Transfer**: Leverages a pre-trained reference atlas to enable flexible and accurate knowledge transfer to new query datasets.
*   **Efficient and Scalable**: Built on PyTorch Lightning for highly efficient model training, with support for advanced strategies like Distributed Data Parallel (DDP).
*   **Advanced Visualization**: Integrates with TensorBoard for real-time monitoring of training loss and UMAP visualizations.

## 🚀 Installation

Get started with MIDAS by setting up a conda environment.

```bash
# 1. Create and activate a new conda environment
conda create -n scmidas python=3.12
conda activate scmidas

# 2. Install MIDAS from PyPI
pip install scmidas
```

## ⚡ Getting Started: A Quick Example

Here is a minimal example to get you started with a mosaic integration task. For more detailed tutorials, please refer to our [documentation](https://scmidas.readthedocs.io/en/latest/).

```python
from scmidas.config import load_config
from scmidas.model import MIDAS
import lightning as L

# 1. Configure and initialize the MIDAS model
# The configuration file allows you to specify modalities, layers, and other parameters.
configs = load_config()

# 2. Load your mosaic dataset
# The input should be an AnnData object where modalities are stored.
# Different batches can have different combinations of modalities.
model = MIDAS.configure_data_from_dir(configs, 'path/to/your/data', transform={'atac':'binarize'})

# 3. Train the model on your data
trainer = L.Trainer(max_epochs=2000)
trainer.fit(model=model)

# 4. Obtain the integrated and imputed results
# The model returns an AnnData object with a unified latent space 
# and imputed values for the missing modalities.
pred = model.predict()

# 5. Visualize the results
model.get_emb_umap()
```

## 📈 Reproducibility

To reproduce the results from our publication, please visit the `reproducibility` branch of this repository:
[**github.com/labomics/midas/tree/reproducibility**](https://github.com/labomics/midas/tree/reproducibility/)

## 📜 Citation

If you use MIDAS in your research, please cite our paper:

He, Z., Hu, S., Chen, Y. *et al*. Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS. *Nat Biotechnol* (2024). https://doi.org/10.1038/s41587-023-02040-y

```bibtex
@article{he2024mosaic,
  title={Mosaic integration and knowledge transfer of single-cell multimodal data with MIDAS},
  author={He, Zhen and Hu, Shuofeng and Chen, Yaowen and An, Sijing and Zhou, Jiahao and Liu, Runyan and Shi, Junfeng and Wang, Jing and Dong, Guohua and Shi, Jinhui and others},
  journal={Nature Biotechnology},
  pages={1--12},
  year={2024},
  publisher={Nature Publishing Group US New York}
}
```

## 🙌 Contributing

We welcome contributions from the community! If you have a suggestion, bug report, or want to contribute to the code, please feel free to open an issue or submit a pull request.

## 📝 License

MIDAS is available under the [MIT License](https://github.com/labomics/midas/blob/main/LICENSE).
