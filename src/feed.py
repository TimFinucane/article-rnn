from glob import glob
import numpy as np
import tensorflow as tf

from rnn_util import distributions
from rnn_util.encoding import WordEncoder

# Feeding into the training process
def feed_articles( batch_size ):
    articles = []
    for article_name in glob( './data/articles/*.txt' ):
        with open( article_name, 'r', encoding = 'utf-8' ) as article:
            articles.append( encoder.encode( article.read() ) )

    # Sort to ensure minimum padding necessary
    articles = sorted( articles, key = (lambda article: len(article)) )

    def inform_shapes( *shapes ):
        def set_shapes( *inputs ):
            for idx, shape in enumerate( shapes ):
                inputs[idx].set_shape( shape )
            return inputs if len(inputs) > 1 else inputs[0]
        return set_shapes

    dataset = tf.data.Dataset.from_generator( lambda: articles, tf.int32, [None, None] )
    
    dataset = dataset.repeat().padded_batch( batch_size, [-1, -1], encoder.ZERO_CLASS )
    iterator = dataset.make_one_shot_iterator()

    return iterator

# Creating char info for numerical distribution in tensorflow
def create_distribution():
    print("Creating distribution from data/articles")

    dist = distributions.DistributionAnalyser()

    for filename in glob("./data/articles/*"):
        dist.check_file( filename )
    
    dist.save( "./data/distribution.txt" )
    dist.save_char_csv( "./data/char_distribution.txt" )

    print("Distribution created")

if __name__ == '__main__':
    create_distribution()

encoder = WordEncoder( distributions.load_char_distribution( './data/char_distribution.txt' ) )
