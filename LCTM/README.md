# Latent Concept Topic Model (LCTM)
This is an implementation of LCTM [1]. LCTM aims to resolve the data sparsity in short texts by inferring topics based on document-level co-occurrence of *latent concepts*.

# Installation 
## Requirements 
You must have the following already installed on your system.

- C++11
- make
- Eigen

## Input and output 
The following files are required for inputs.

- Initial concept assignment file: i-th row indicates the initial concept assignment of i-th word type. We recommend performing k-means clustering to initialize concept assignment.
- Word embeddings file: 1st row contains #(vocabulary) and #(dimension of word vectors). From the second row, i-th row contains vector representation for (i-1)-th word type.
- Indexed corpus: Each row contains a list of indices of words contained each document.

The software outputs the following file.

- theta: Document-topic distribution
- phi: Topic-concept distribution
- mu: concept vector
- noise: noise for each concept

## Quick start
Type `make; sh run.sh` 

The codes will be compiled and run on dataset in the directory `input/sample` with default parameters.
To modify the parameters or path to the dataset directory, edit the corresponding part in run.sh.

## Reference ##
[1] Weihua Hu and Jun'ichi Tsujii. A Latent Concept Topic Model for Robust Topic Inference Using Word Embeddings. 2016. Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics (ACL-16 short paper)