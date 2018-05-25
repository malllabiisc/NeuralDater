## Dating Documents using Graph Convolution Networks

Source code and dataset for [ACL 2018](http://acl2018.org) paper: [Document Dating using Graph Convolution Networks](http://malllabiisc.github.io/publications/papers/neuraldater_acl18.pdf).

<img src="https://raw.githubusercontent.com/malllabiisc/NeuralDater/master/overview.png" alt="https://raw.githubusercontent.com/malllabiisc/NeuralDater/master/overview.png">

### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.


### Dataset:

* Download processed version of [NYT](https://drive.google.com/file/d/1wqQRFeA1ESAOJqrwUNakfa77n_S9cmBi/view?usp=sharing) and [APW](https://drive.google.com/open?id=1tll04ZBooB3Mohm6It-v8MBcjMCC3Y1w) section of  [Gigaword Corpus, 5th ed](https://catalog.ldc.upenn.edu/ldc2011t07).
* After downloading, unzip the `.pkl` file in `data` directory.

### Usage:

* After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

* `neural_dater.py` contains the TensorFlow (1.x) implementation of the Neural dater. To start training run: 

  ```shell
  python neural_dater.py -data data/nyt_processed_data.pkl -class 10 -name test_run
  ```

  * `-class` denotes the number of classes in datasets,  `10` for NYT and `16` for APW.
  * `-name` is arbitrary name for the run.

* We recommend to use tab size `8` for viewing `*.py` files.

