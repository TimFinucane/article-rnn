# Article Generator

This code is used for training an RNN on large amounts of coherent text data, where
each 'article' is also too large to train on in one run.

The algorithm usees characters as its basis, rather than a corpus, however
does a 2 pass run where related characters are converted into words through an
initial RNN before then being run through a word-level RNN.

The code supports saving and restoring from a model in the training function.
Models (saved through tf.train.Saver) are placed by default into ./save