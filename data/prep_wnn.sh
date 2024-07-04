#!/bin/bash

Rscript preprocess/rna+adt_wnn.R
Rscript preprocess/combine_subsets.R --task wnn_demo 