import tensorflow as tf

NUM_LAYERS = 3
HIDDEN_UNITS = 1024

def state_shape( batch_size ):
    return [NUM_LAYERS, batch_size, HIDDEN_UNITS]

with tf.variable_scope( 'char_lstm' ) as model_scope:
    pass

def model( state, full_words, char_embedding:int, num_guesses: int = 4, training = True ):
    '''
    Expects words as tensor of dimensions [batches, words, characters],
    padded appropriately.

    When training = True, will not generate from the final word, but
    instead use that for prediction of it's final word in the char-rnn.
    
    # Returns
    A tuple output of (guesses, guess_p_logits), state.
    guesses: [batches, words, guesses, characters, one_hot]
    guess_probabilities: [batches, words, guesses]. Confidence for each of the num guesses
    '''
    CW_LAYERS = 3
    CW_UNITS = 512
    # chars -> chars RNN. Many-To-Many. Uses state to interpret words
    ctc_rnn = tf.contrib.cudnn_rnn.CudnnRNNRelu( CW_LAYERS, CW_UNITS )

    inputs = full_words
    with tf.variable_scope( model_scope, reuse = not training ):
        with tf.variable_scope( 'chars_to_words' ):
            # Reform to [characters (sequence), batches * words (batches), embedding (rnn inputs)]
            chars = full_words[:, :-1, :] if training else full_words
            chars = tf.transpose( tf.one_hot( chars, char_embedding ), [2, 0, 1, 3] )
            chars = tf.reshape( chars, [tf.shape( chars )[0], -1, char_embedding] )

            ctc_predictions, ctw_state = ctc_rnn( chars, training = training )
            # Take middle layer of state for word meanings [batches * words, embedding of CW_UNITS]
            words_input_embeddings = tf.reshape( ctw_state[0][1], [tf.shape( full_words )[0], -1, CW_UNITS] )
            words = tf.layers.batch_normalization( ctw_state[0][1], training = training )

            if training: # Ensure the char_rnn correctly predicts the word with a loss
                # ctc_predictions = [characters, batches * words, embedding]
                ctc_predictions = tf.layers.conv1d( ctc_predictions, char_embedding, 1, name = 'ctc_embedding_to_onehot' )
                ctc_predictions = tf.reshape( ctc_predictions, [tf.shape( inputs )[2], tf.shape( inputs )[0], -1, char_embedding] )
                ctc_predictions = tf.transpose( ctc_predictions, [1, 2, 0, 3] )
                ctw_loss = tf.reduce_mean( tf.nn.sparse_softmax_cross_entropy_with_logits( logits = ctc_predictions[..., :-1, :], labels = full_words[:, :-1, 1:] ) )
            
        with tf.variable_scope( 'inner' ):
            # Reform words to [words (sequence), batches, embedding]
            words = tf.reshape( words, [tf.shape( inputs )[0], -1, CW_UNITS] )
            words = tf.transpose( words, [1, 0, 2] )

            # word -> word GRU. Many-To-Many. Generates a guessed next word from the current one
            word_to_word = tf.contrib.cudnn_rnn.CudnnGRU( NUM_LAYERS, HIDDEN_UNITS )
            wtw_output, output_state = word_to_word( words, (state,), training = training )
            wtw_output = tf.transpose( wtw_output, [1, 0, 2] )
            wtw_output.set_shape( [None, None, HIDDEN_UNITS] )
            # Make 4 guesses of potential words
            guesses = []
            for i in range( num_guesses ):
                guesses.append( tf.layers.conv1d( wtw_output, CW_UNITS, 1, name = 'guess_{:}'.format( i ) ) )
            words_with_guesses = tf.stack( guesses, 2 ) # [batches, words, guesses(num_guesses), embeddings(CW_UNITS)]
            words_with_guesses = tf.layers.batch_normalization( words_with_guesses, training = training )

            guess_choice_logits = tf.layers.conv1d( wtw_output, num_guesses, 1, name = 'guess_certainty' ) # [batches, words, guesses (num_guesses)]

        with tf.variable_scope( 'words_to_chars' ) as wtc_scope:
            if training:
                guess_choice = tf.argmax( guess_choice_logits, -1 )
                # guess_choice is now an index into each word, with shape (batches, words)
                # Use it to get the desired word. Not a great way but gather and gather_nd cant be used for this purpose.
                words_output_embeddings = tf.reduce_sum( words_with_guesses * tf.expand_dims( tf.one_hot( guess_choice, num_guesses ), -1 ), axis = 2 )
                wtw_loss = tf.reduce_mean( tf.abs( words_output_embeddings[:, :-1] - words_input_embeddings[:, 1:] ), axis = [1, 2] )

            # Create expected inputs of all the real characters for each word. First char is zero, then each
            # following character is the real character of the word
            if training:
                # Reform to [(batches word guesses), embeddings]
                wtc_state = tf.reshape( words_with_guesses, [1, -1, CW_UNITS] )
                wtc_state = tf.concat( (tf.zeros_like( wtc_state ), wtc_state, tf.zeros_like( wtc_state )), 0 )

                # Provide the characters of what the next word should be to help word2char along. First char is blank to signify
                #  blankness.
                wtc_inputs = tf.one_hot( full_words[:, 1:, :-1], char_embedding ) # chars from input
                wtc_inputs = tf.concat( (tf.zeros_like( wtc_inputs[:, :, 0:1, :] ), wtc_inputs), 2 ) # blank first char
                wtc_inputs = tf.tile( tf.expand_dims( wtc_inputs, 2 ), [1, 1, num_guesses, 1, 1] ) # repeat num_guesses times to match the inner words input
                wtc_inputs = tf.reshape( wtc_inputs, [-1, tf.shape( inputs )[2], char_embedding] ) # [batch x words x guesses, chars, embeddings]
                wtc_inputs = tf.transpose( wtc_inputs, [1, 0, 2] ) # [chars, batches x words x guesses, embeddings]

                chars_with_guesses, _ = ctc_rnn( wtc_inputs, (wtc_state,), training = training )
                # [characters, (batches words guesses), embeddings]
                chars_with_guesses = tf.layers.conv1d( chars_with_guesses, char_embedding, 1, name = 'char_embeddings_to_onehot' ) # Convert embeddings to char one-hot
                chars_with_guesses = tf.reshape( chars_with_guesses, [tf.shape( inputs )[2], tf.shape( inputs )[0], -1, num_guesses, char_embedding] )
                chars_with_guesses = tf.transpose( chars_with_guesses, [1, 2, 3, 0, 4] ) # [batches, words, guesses (num_guesses), characters, one_hot]

                return (chars_with_guesses, guess_choice_logits), output_state[0], (ctw_loss, wtw_loss)
            else:
                # Choose the word to continue with
                guess_choice = tf.multinomial( tf.reshape( guess_choice_logits, [-1, num_guesses] ), 1, output_dtype = tf.int32 )[:, 0]
                # guess_choice is now an index into each word, with shape (batches x words)
                words_with_guesses = tf.reshape( words_with_guesses, [-1, num_guesses, CW_UNITS] )
                guess_choice = tf.stack( [tf.range( tf.shape( guess_choice )[0], dtype = tf.int32 ), guess_choice], -1 )
                words = tf.gather_nd( words_with_guesses, guess_choice )
                # words should have shape [batches x words, embeddings]
                wtc_state = tf.reshape( words, [1, -1, CW_UNITS] )
                wtc_state = tf.concat( (tf.zeros_like( wtc_state ), wtc_state, tf.zeros_like( wtc_state )), 0 )

                zero_char_input = tf.zeros( [1, tf.shape( words )[0], char_embedding] ) # [1, batches x words, one_hot]
                first_char, (wtc_state,) = ctc_rnn( zero_char_input, (wtc_state,), training = False )
                first_char = tf.layers.conv1d( first_char, char_embedding, 1, name = 'char_embeddings_to_onehot', reuse = True ) # Convert embeddings to char one-hot
                # Turn into a next character - uses nondeterministic prediction instead of really boring argmax
                first_char = tf.multinomial( first_char[0], 1, output_dtype = tf.int32 )[:, 0]

                def body( i, char_arr, prev_wtc_state ):
                    char = tf.reshape( tf.one_hot( char_arr.read( i - 1 ), char_embedding ), [-1, char_embedding] )
                    with tf.variable_scope( wtc_scope, reuse = True ):
                        next_char, (next_wtc_state,) = ctc_rnn( tf.expand_dims( char, 0 ), (prev_wtc_state,), training = False )
                        next_char = tf.layers.conv1d( next_char, char_embedding, 1, name = 'char_embeddings_to_onehot', reuse = True ) # Convert embeddings to char one-hot
                    next_char = tf.squeeze( next_char, [0] ) # [batches x words, embeddings]
                    # Turn into a next character - uses nondeterministic prediction instead of really boring argmax
                    next_char = tf.multinomial( next_char, 1, output_dtype = tf.int32 )[:, 0]
                    char_arr = char_arr.write( i, next_char ) # Write to array, remember to take return arg

                    return i + 1, char_arr, next_wtc_state
                def cond( i, _a, _b ):
                    return tf.less( i, tf.shape( inputs )[2] )

                char_array = tf.TensorArray( tf.int32, size = 1, dynamic_size = True, clear_after_read = False )
                char_array = char_array.write( 0, first_char )

                _, char_array, _ = tf.while_loop( cond, body, [tf.constant( 1 ), char_array, wtc_state], back_prop = False )

                chars = char_array.stack() # [characters, batches x words]
                chars = tf.reshape( tf.transpose( chars, [1, 0] ), [tf.shape( full_words )[0], -1, tf.shape( full_words )[2]] )
                
                return chars, output_state[0]
