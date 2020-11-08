#!/usr/bin/env python3
import urllib.parse
import urllib.request

if __name__ == '__main__':

    url = 'https://www.uniprot.org/uniprot/?'

    # Get UniProt table with protein sequences
    params = {
        'query': 'reviewed:yes',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params)
    urllib.request.urlretrieve(url + query,
        '../../data/raw/uniprot_table.txt')

    # Get table with UniProt keywords
    urllib.request.urlretrieve('https://www.uniprot.org/keywords/?query=*&'
                               'format=tab&force=true&columns=id',
                               '../../data/raw/uniprot_keywords.txt')