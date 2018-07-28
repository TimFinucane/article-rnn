import os
from itertools import count
import numpy as np
import tensorflow as tf

from feed import feed_articles, encoder
from model import model, state_shape

def write_to_file( filename, text, overwrite = False ):
    with open( filename, 'w' if overwrite else 'a', encoding = 'utf-8' ) as file:
        file.write( '\n'.join( text ) + '\n' )

def generate( initial_word, batch_size, max_length = 1000 ):
    def body( i, state, char_array ):
        model_input = tf.expand_dims( char_array.read( i - 1 ), 1 )
        # Pass in previous input get next output
        model_output, next_state = model( state, model_input, encoder.CLASSES, training = False )
        # Place character into char array for next go.
        char_array = char_array.write( i, tf.squeeze( model_output, [1] ) )
        return i + 1, next_state, char_array

    def cond( i, _state, char_array ):
        return tf.logical_and(
            tf.reduce_any( tf.not_equal( char_array.read( i - 1 ), encoder.ZERO_CLASS ) ),
            tf.less( i, max_length )
        )

    # Create an array to store extrapolated segments
    char_array = tf.TensorArray( tf.int32, size = 1, dynamic_size = True, element_shape = [batch_size, None], clear_after_read = False )
    char_array = char_array.write( 0, initial_word )

    # Produce the rest of the segments
    _, _, char_array = tf.while_loop( cond, body, [tf.constant( 1 ), tf.zeros( state_shape( batch_size ) ), char_array], back_prop = False )

    char_array = char_array.stack() # Stack them up into a single tensor, [words, batch, chars]
    char_array = tf.transpose( char_array, [1, 0, 2] ) # [batch, words, chars]

    return char_array

def train():
    BATCH_SIZE = 8
    TRAIN_LENGTH = 64

    global_step = tf.Variable( 0, name = 'global_step' )

    article_iterator = feed_articles( BATCH_SIZE )

    articles = tf.get_variable( 'current_articles', shape = [BATCH_SIZE, 0, 0], validate_shape = False, dtype = tf.int32, trainable = False )
    pos = tf.get_variable( 'article_pos', dtype = tf.int32, trainable = False, initializer = tf.constant( 0 ) )
    state = tf.get_variable( 'state', state_shape( BATCH_SIZE ) )

    with tf.variable_scope( 'ops' ):
        def reset_op():
            next_batch = article_iterator.get_next()
            # Pad to ensure that we can train with a full train_length
            middle_pad = (TRAIN_LENGTH + 1) - (tf.shape( next_batch )[1] % TRAIN_LENGTH)
            next_batch = tf.pad( next_batch, [[0, 0], [0, middle_pad], [0, 0]], constant_values = encoder.ZERO_CLASS )

            return tf.group(
                # Move to next audio batch. Repeat first segment for purposes of training first part.
                tf.assign( articles, next_batch, validate_shape = False ),
                # Reset the state to all zeros
                pos.assign( 0 ),
                state.assign( tf.zeros( state_shape( BATCH_SIZE ) ) )
            )

        training_update = tf.cond( tf.greater_equal( pos.assign( pos + TRAIN_LENGTH ), tf.shape( articles )[1] - 1 ), reset_op, lambda: tf.no_op() )

    with tf.variable_scope( 'model' ):
        (guesses, guess_prob), next_state, embed_cost = model( state, articles[:, pos:(pos + TRAIN_LENGTH + 1)], encoder.CLASSES )
        guess_prob = tf.expand_dims( tf.expand_dims( tf.nn.softmax( guess_prob ), -1 ), -1 ) # Shape for broadcasting
        predicted_logits = tf.reduce_logsumexp( guess_prob * guesses, 2 )
        
        # Have diminishing cost the closer you get to the first char. This is to let the model be less precise at the start
        cost = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = predicted_logits, labels = articles[:, pos + 1:(pos + 1 + TRAIN_LENGTH)] ) )
        cost += tf.reduce_mean( embed_cost[1] ) * 0.2

        # Have some output stuff
        predicted_classes = tf.concat( (articles[:, pos:(pos + 1)], tf.argmax( predicted_logits, -1, output_type = tf.int32 )), 1 )
        # A generator that creates a full article
        generated_classes = generate( articles[0:1, 0], 1, max_length = 1000 )

    with tf.variable_scope( 'train' ):
        learning_rate = tf.Variable( 1e-4, trainable = False )

        with tf.control_dependencies( [tf.assign( state, next_state ), global_step.assign( global_step + 1 ), *tf.get_collection( tf.GraphKeys.UPDATE_OPS )] ):
            trainer = tf.train.AdamOptimizer( learning_rate ).minimize( cost )

    with tf.Session( config = tf.ConfigProto( gpu_options = tf.GPUOptions( per_process_gpu_memory_fraction = 0.75 ) ) ) as session:
        session.run( tf.global_variables_initializer() )

        saver = tf.train.Saver()
        saver.export_meta_graph( './save/model.meta' )
        # saver.restore( session, './save/model' )

        try:
            while True:
                i, _ = session.run( [global_step, training_update] )
                _cost, _predictions, _ = session.run( [cost, predicted_classes, trainer] )

                print( '{:05d}: {:6.4f}, "{:}"'.format( i, _cost, encoder.decode( _predictions[0] ) ) )

                if i % 1000 == 0:
                    write_to_file( './data/generated_{:04d}.txt'.format( i // 1000 ),
                                   [encoder.decode( article ) for article in session.run( generated_classes )],
                                   overwrite = True )
                    saver.save( session, './save/model' )
                #if i % 500 == 0 and i <= 10000:
                #    session.run( lr_adjust )
        except KeyboardInterrupt:
            print( 'saving' )

        saver.save( session, './save/model' )
if __name__ == '__main__':
    train()
