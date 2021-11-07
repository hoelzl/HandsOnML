# Hands-On Machine Learning

My version of the code for the second edition of the book "Hands-On Machine
Learning with Scikit-Learn, Keras and TensorFlow" by Aurélien Geron.

The original code for the book and the data files can be found at the book's
[GitHub repository](https://github.com/ageron/handson-ml2).

Just like the original code, the code in this repository is licensed under the
Apache 2.0 license.

My current plan is to use PyTorch/Skorch for the Deep Learning examples, but
we'll see how this turns out.

You can set up a conda environment for this project using the included
`ml.yaml` file. This environment is what I use as a baseline for my machine
learning projects and contains more than is strictly necessary for the code
in this repository. Create it with

```shell
conda env create --file ml.yaml
```

or, if you have [mamba](https://github.com/mamba-org/mamba) installed (which I
recommend since it greatly speeds up environment management with conda):

```shell
mamba env create --file ml.yaml
```


