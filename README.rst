==============================================
MA-sLDAr -- Multi-Annotator Supervised LDA for regression
==============================================

`MA-sLDAr` is a C++ implementation of the supervised topic models with response variables provided by multiple annotators with different levels of expertise, as proposed in:

* `Rodrigues, F., Lourenço, M, Ribeiro, B, Pereira, F. Learning Supervised Topic Models for Classification and Regression from Crowds. IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI), 2017 <http://www.fprodrigues.com/publications/learning-supervised-topic-models-for-classification-and-regression-from-crowds/>`_.

A version of this model for classification tasks is available `here <https://github.com/fmpr/MA-sLDAc>`_.

Sample multiple-annotator data using the we8there dataset is provided `here <http://www.fprodrigues.com/we8there.tar.gz>`_. More datasets are available `here <http://www.fprodrigues.com/ma-sldar-multi-annotator-supervised-lda-for-regression/>`_. 

`MA-sLDAr` is open source software released under the `GNU LGPL license <http://www.gnu.org/licenses/lgpl.html>`_.
Copyright (c) 2016-now Filipe Rodrigues

Compiling
------------

Type "make" in a shell. 

Please note that this code requires the Gnu Scientific Library, http://www.gnu.org/software/gsl/

Estimation
------------

Usage:: 

    ./maslda est [data] [answers] [settings] [alpha] [tau] [k] [random/seeded/model_path] [seed] [directory]

Data format:

* [data] is a file where each line is of the form: [M] [term_1]:[count] [term_2]:[count] ...  [term_N]:[count], where [M] is the number of unique terms in the document, and the [count] associated with each term is how many times that term appeared in the document. 
* [answers] is a file where each line contains the labels of the different annotators (separated by a white space) for [data]. Each column therefore corresponds to all the answers of an annotator. The labels must be 0, 1, ..., C-1, if we have C classes. Missing answers are encoded using "-1".

Example:: 

    ./maslda est ../MovieReviews/data_train_amt.txt ../MovieReviews/answers.txt settings.txt 1 0.1 20 random 1 output

Inference
------------

Usage:: 

    ./maslda inf [data] [label] [settings] [model] [directory]

Data format: 

* [labels] is a file where each line is the corresponding true label for [data].

Example:: 

    ./maslda inf ../MovieReviews/data_test.txt ../MovieReviews/labels_test.txt settings.txt output/final.model output

Settings
------------

The settings file specifies the following parameters:

* "L2 penalty" controls the strength of the L2 regularization.
* "labels train file" is a file with the true target variables for the training documents. If a valid file is provided, it will be use to compute and report error statistics during the model estimation.
* "annotators quality file" is a file with the true biases and variances of the multiple annotators. If a valid file is provided, it will be use to compute and report error statistics during the model estimation.
* "lambda smoother" defines the values of the laplace smoothers used when estimating pi and lambda respectively.

