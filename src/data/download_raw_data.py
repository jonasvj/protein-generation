#!/usr/bin/env python3
import os
import subprocess
import urllib.parse
import urllib.request

if __name__ == '__main__':

    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()
    
    url = 'https://www.uniprot.org/'

    # Get UniProt table with protein sequences
    params = {
        'query': 'reviewed:yes',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params)
    
    urllib.request.urlretrieve(
        url + 'uniprot/?' + query,
        os.path.join(repo_dir, 'data/raw/uniprot_table.txt'))

    # Get table with UniProt keywords
    urllib.request.urlretrieve(
        url + 'keywords/?' + 'query=*&format=tab&force=true&columns=id''
        os.path.join(repo_dir, 'data/raw/uniprot_keywords.txt'))
    