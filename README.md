# MIDAS

A deep generative model for **M**osaic **I**ntegration and knowle**D**ge tr**A**n**S**fer of single-cell multimodal data.

## File/folder instruction

| File/directory | Description                                                 |
| -------------- | ----------------------------------------------------------- |
| `comparison/`  | Scripts for algorithm comparison and qualitative evaluation |
| `configs/`     | Dataset configuration and MIDAS model configuration         |
| `eval/`        | Scripts for quantitative evaluation                         |
| `functions/`   | PyTorch functions for MIDAS                                 |
| `modules/`     | PyTorch models and dataloader for MIDAS                     |
| `preprocess/`  | Scripts for data preprocessing                              |
| `utils/`       | Commonly used functions                                     |
| `README.md`    | This file                                                   |
| `run.py`       | Script for MIDAS training and inference                     |

## Requirements

*   PyTorch 1.12.0
*   Seurat 4.1.0
*   Signac 1.6.0
