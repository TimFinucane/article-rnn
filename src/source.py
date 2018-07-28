from urllib.request import Request, urlopen
from bs4 import BeautifulSoup

urls = [
]

for idx, url in enumerate(urls):
    html = urlopen( Request( url, headers={'User-Agent' : "Non-Magic Browser"} ) )
    soup = BeautifulSoup( html, "html5lib" )

    # REPLACE THIS WITH RELEVANT SOURCING FOR YOUR ARTICLES
    soup.find_all( 'div' )
    soup = soup.find( 'span', attrs = { 'class': 'cb-itemprop', 'itemprop': 'reviewBody' } )

    [div.decompose() for div in soup.find_all( 'div' )]
    # ===

    text = soup.get_text()

    with open( './data/articles/' + url.split( '/' )[-1] + '.txt', 'w', encoding = 'utf-8' ) as article_file:
        article_file.write( text )
    
    print( str(idx) + '. completed ' + url )