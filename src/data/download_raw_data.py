#!/usr/bin/env python3
import os
import subprocess
import urllib.parse
import urllib.request

if __name__ == '__main__':
    
    repo_dir = subprocess.run(
        ['git', 'rev-parse', '--show-toplevel'],
        stdout=subprocess.PIPE).stdout.decode().strip()
    """
    url = 'https://www.uniprot.org/'

    # Get table with UniProt keywords
    urllib.request.urlretrieve(
        url + 'keywords/?' + 'query=*&format=tab&force=true&columns=id',
        os.path.join(repo_dir, 'data/raw/uniprot_keywords.txt'))


    # Get UniProt table with protein sequences (insulin in eukaryotes)
    params_insulin = {
        'query': 'insulin taxonomy:"Eukaryota [2759]" AND reviewed:yes',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params_insulin)
    
    urllib.request.urlretrieve(
        url + 'uniprot/?' + query,
        os.path.join(repo_dir, 'data/raw/uniprot_table_insulin.txt'))

    # Get UniProt table with protein sequences (not insulin in eukaryotes)
    params_not_insulin = {
        'query': 'NOT insulin taxonomy:"Eukaryota [2759]" AND reviewed:yes',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params_not_insulin)
    
    urllib.request.urlretrieve(
        url + 'uniprot/?' + query,
        os.path.join(repo_dir, 'data/raw/uniprot_table_not_insulin.txt'))
    """
    #Add 'insulin related' column to both files
    insulin_data = open(os.path.join(repo_dir, 'data/raw/uniprot_table_insulin.txt'), 'r')
    insulin_edited = open(os.path.join(repo_dir, 'data/raw/insulin_table.txt'), 'w')

    first = True
    for line in insulin_data:
        if first == True:
            insulin_edited.write(line[:-1] + "\tRelated to insulin\n")
            first = False
        else:
            insulin_edited.write(line[:-1] + "\tYes\n")
        
    insulin_data.close()
    insulin_edited.close()
    
    not_insulin_data = open(os.path.join(repo_dir, 'data/raw/uniprot_table_not_insulin.txt'), 'r')
    not_insulin_edited = open(os.path.join(repo_dir, 'data/raw/not_insulin_table.txt'), 'w')

    first = True
    for line in not_insulin_data:
        if first == True:
            first = False
        else:
            not_insulin_edited.write(line[:-1] + "\tNo\n")
        
    not_insulin_data.close()
    not_insulin_edited.close()

    insulin_edited = open(os.path.join(repo_dir, 'data/raw/insulin_table.txt'), 'r')
    not_insulin_edited = open(os.path.join(repo_dir, 'data/raw/not_insulin_table.txt'), 'r')
    data = open(os.path.join(repo_dir, 'data/raw/uniprot_table_data.txt'), 'w')

    for file in [insulin_edited, not_insulin_edited]:
        data.write(file.read())

    data.close()
    insulin_edited.close()
    not_insulin_edited.close()