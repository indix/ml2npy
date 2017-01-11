# ml2npy - Export spark ml SparseVectors as numpy csr matrix

[![Build Status](https://app.snap-ci.com/indix/ml2npy/branch/master/build_image)](https://app.snap-ci.com/indix/ml2npy/branch/master)

The aim of this project is to provide that tools that efficiently implement the components that are required for large scale text mining. 

The idea for this project came out from experience,
  1. Most of time it is data preprocessing that is expensive and demanding
  2. Distributed algorithm implementations are not still as effective as Multicore/sequential implementations.

This project intends to leverage the best of both worlds. In case of text mining, a traditional powerful approach is to use [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) as numerical representation of the document. This enables a vareity of machine learning techniques to be readily applied on the data. Converting a document in to TF-IDF or any other numerical format is compute intensive and once a numerical representation is available, we could try out various algorithms and models on the preprocessed data. 

Numerical representation of text tends to be very sparse. By choosing [sparse matrix formats](https://en.wikipedia.org/wiki/Sparse_matrix) to save this data, we could save memory and disk usage. ml2npy provides tools and utilities to load a large corpus of text and save its numerical respresentation as CSR Matrix in [numpy format](https://docs.scipy.org/doc/numpy/neps/npy-format.html)

### Why Npy format?

Python and scikit-learn ecosystem has made machine learning a lot more accessible. By being able to load data in to python, means a lot of algorithms could be easily applied.




