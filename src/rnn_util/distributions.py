from glob import glob
import re
import csv

from collections import defaultdict
from operator import itemgetter

def load_char_distribution( filename ):
    chars = []
    with open( filename, 'r', encoding = 'utf-8' ) as csv_file:
        reader = csv.reader( csv_file )
        for row in reader:
            chars.append( row[0] )
    return chars

class DistributionAnalyser():
    '''
    Checks the char and word distribution of a series of files or text
    '''
    def __init__( self ):
        self.words = defaultdict(int)
        self.characters = defaultdict(int)

    def check_file( self, filename ):
        with open( filename, 'r', encoding = 'utf-8' ) as article:
            text = article.read()

            for c in text:
                self.characters[c] += 1

            text = re.split( r"([a-zA-Z0-9]+[-’']?[a-zA-Z0-9]+|\w|\W)", text )
            for item in filter( lambda word: word, text ):
                self.words[item] += 1

    def save_char_csv( self, filename = './chars.txt' ):
        with open( filename, 'w', encoding = 'utf-8', newline = '' ) as csv_file:
            writer = csv.writer( csv_file )
            for char in sorted( self.characters ):
                writer.writerow( [char] )


    def save( self, filename = './distribution.txt' ):
        with open( filename, 'w', encoding = 'utf-8' ) as dist_file:
            dist_file.write( str(len(self.characters)) + ' characters\n' )
            dist_file.write( '"' + '", "'.join( sorted( self.characters.keys() ) ) + '"\n' )
            for char, occurence in sorted( self.characters.items(), key=itemgetter( 1 ), reverse = True ):
                dist_file.write( str(char) + ', ' + str(occurence) + '\n' )

            dist_file.write( '\n\n' )

            dist_file.write( str(len(self.words)) + ' words\n' )
            dist_file.write( '"' + '", "'.join( sorted( self.words.keys() ) ) + '"\n' )
            for char, occurence in sorted( self.words.items(), key=itemgetter( 1 ), reverse = True ):
                dist_file.write( str(char) + ', ' + str(occurence) + '\n' )
