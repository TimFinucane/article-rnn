import re
import numpy as np

class WordEncoder:
    def __init__( self, chars ):
        self.chars = chars
        self.chars.append( '\0' )
        self.ZERO_CLASS = len(self.chars) - 1
        self.CLASSES = len(self.chars)

    def encode( self, string ):
        '''
        Converts a string into a series of text classes
        '''
        words = re.split( r"(\w+[’']?\w+|\w|\W)", string )
        words = filter( lambda word: word, words )

        max_length = 0
        encodings = []
        for word in words:
            encoded_word = [self.chars.index( c ) for c in word]
            encodings.append( encoded_word )
            max_length = max( (len(encoded_word), max_length) )

        encodings = np.array( [np.pad( word, [(0, max_length - len(word))], 'constant', constant_values = self.ZERO_CLASS ) for word in encodings] )

        return encodings
    
    def decode( self, classes ):
        '''
        Converts encoded classes into a string
        '''
        return ''.join( self.chars[i] for i in classes.flatten() if i != self.ZERO_CLASS )

    