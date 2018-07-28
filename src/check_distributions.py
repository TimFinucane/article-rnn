from glob import glob
import re

from collections import defaultdict
from operator import itemgetter
from matplotlib import pyplot as plt

words = defaultdict(int)
characters = defaultdict(int)

for filename in glob( './data/articles/*.txt' ):
    with open( filename, 'r', encoding = 'utf-8' ) as article:
        text = article.read()

        for c in text:
            characters[c] += 1

        text = re.split( r"(\w+[-’']?\w+|\w|\W)", text )
        for item in filter( lambda word: word, text ):
            words[item] += 1

with open( './data/distribution.txt', 'w', encoding = 'utf-8' ) as dist_file:
    dist_file.write( str(len(characters)) + ' characters\n' )
    dist_file.write( '"' + '", "'.join( sorted( characters.keys() ) ) + '"\n' )
    for char, occurence in sorted( characters.items(), key=itemgetter( 1 ), reverse = True ):
        dist_file.write( str(char) + ', ' + str(occurence) + '\n' )

    dist_file.write( '\n\n' )

    dist_file.write( str(len(words)) + ' words\n' )
    dist_file.write( '"' + '", "'.join( sorted( words.keys() ) ) + '"\n' )
    for char, occurence in sorted( words.items(), key=itemgetter( 1 ), reverse = True ):
        dist_file.write( str(char) + ', ' + str(occurence) + '\n' )
