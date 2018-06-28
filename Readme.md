## Dating Documents using Graph Convolution Networks

Source code and dataset for [ACL 2018](http://acl2018.org) paper: [Document Dating using Graph Convolution Networks](http://malllabiisc.github.io/publications/papers/neuraldater_acl18.pdf).

![](https://raw.githubusercontent.com/malllabiisc/NeuralDater/master/overview.png)
*Overview of NeuralDater (proposed method). NeuralDater exploits syntactic and temporal structure in a document to learn effective representation, which in turn are used to predict the document time. NeuralDater uses a Bi-directional LSTM (Bi-LSTM), two Graph Convolution Networks (GCN) – one over the dependency tree and the other over the document’s temporal graph – along with a softmax classifier, all trained end-to-end jointly. Please refer paper for more details.*
### Dependencies

* Compatible with TensorFlow 1.x and Python 3.x.
* Dependencies can be installed using `requirements.txt`.


### Dataset:

* Download the processed version (includes dependency and temporal graphs of each document) of [NYT](https://drive.google.com/file/d/1wqQRFeA1ESAOJqrwUNakfa77n_S9cmBi/view?usp=sharing) and [APW](https://drive.google.com/open?id=1tll04ZBooB3Mohm6It-v8MBcjMCC3Y1w) datasets.
* Unzip the `.pkl` file in `data` directory.
* Documents are originally taken from NYT and APW section of [Gigaword Corpus, 5th ed](https://catalog.ldc.upenn.edu/ldc2011t07).

### Usage:

* After installing python dependencies from `requirements.txt`, execute `sh setup.sh` for downloading GloVe embeddings.

* `neural_dater.py` contains TensorFlow (1.x) based implementation of the NeuralDater (proposed method). 
* To start training: 
  ```shell
  python neural_dater.py -data data/nyt_processed_data.pkl -class 10 -name test_run
  ```

  * `-class` denotes the number of classes in datasets,  `10` for NYT and `16` for APW.
  * `-name` is arbitrary name for the run.

