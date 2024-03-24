# Tutorial Setup


### Python Version: 3.10

To participate in this tutorial, it is recommended you use python 3.10.
An easy way to install this is by installing anaconda and creating a new environment with python 3.10.
It should be noted that this isn't a strict requirement, but it is recommended.
Furthermore, it is recommended to create a virtual environment in the root project directory.

Anaconda can be installed here: https://docs.anaconda.com/free/anaconda/install/index.html


### Python Libraries:

You can install the python dependencies by running the following command
in your python environment:

```bash
pip3 install keras-nlp==0.6.1
pip3 install tensorflow==2.14
pip3 install tensorflow-addons
pip3 install matplotlib
```


It is important that you install tensorflow 2.14 after keras-nlp 0.6.1, as keras-nlp will install a more recent
version of tensorflow. This may cause issues during the tutorial if you don't have the correct version of tensorflow (2.14) installed.