#!/usr/bin/env python3
import os
import subprocess
import urllib.parse
import urllib.request
from src.utils import get_repo_dir

if __name__ == '__main__':
    
    repo_dir = get_repo_dir()
    
    url = 'https://www.uniprot.org/'

    # Get table with UniProt keywords
    urllib.request.urlretrieve(
        url + 'keywords/?' + 'query=*&format=tab&force=true&columns=id',
        os.path.join(repo_dir, 'data/raw/uniprot_keywords.txt'))

    # Get UniProt table with protein sequences (insulin in eukaryotes)
    params_insulin = {
        'query': 'database:(type:pfam pf00049) taxonomy:"Eukaryota [2759]"',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params_insulin)
    
    urllib.request.urlretrieve(
        url + 'uniprot/?' + query,
        os.path.join(repo_dir, 'data/raw/uniprot_table_insulin.txt'))

    # Get UniProt table with protein sequences (not insulin in eukaryotes)
    params_not_insulin = {
        'query': 'NOT database:(type:pfam pf00049) taxonomy:"Eukaryota [2759]" AND reviewed:yes',
        'format': 'tab',
        'columns': 'id,entry name,protein names,organism-id,'
                   'keywords,database(Pfam),sequence'}

    query = urllib.parse.urlencode(params_not_insulin)
    
    urllib.request.urlretrieve(
        url + 'uniprot/?' + query,
        os.path.join(repo_dir, 'data/raw/uniprot_table_not_insulin.txt'))
    
    # Add 'Insulin' column to both files
    insulin_data = open(os.path.join(repo_dir,
        'data/raw/uniprot_table_insulin.txt'), 'r')
    insulin_edited = open(os.path.join(repo_dir,
        'data/interim/insulin_table.txt'), 'w')

    first = True
    for line in insulin_data:
        if first == True:
            insulin_edited.write(line[:-1] + "\tInsulin\n")
            first = False
        else:
            insulin_edited.write(line[:-1] + "\tYes\n")
        
    insulin_data.close()
    insulin_edited.close()
    
    not_insulin_data = open(os.path.join(repo_dir,
        'data/raw/uniprot_table_not_insulin.txt'), 'r')
    not_insulin_edited = open(os.path.join(repo_dir,
        'data/interim/not_insulin_table.txt'), 'w')

    first = True
    for line in not_insulin_data:
        if first == True:
            first = False
        else:
            not_insulin_edited.write(line[:-1] + "\tNo\n")
        
    not_insulin_data.close()
    not_insulin_edited.close()

    insulin_edited = open(os.path.join(repo_dir,
        'data/interim/insulin_table.txt'), 'r')
    not_insulin_edited = open(os.path.join(repo_dir,
        'data/interim/not_insulin_table.txt'), 'r')
    data = open(os.path.join(repo_dir,
        'data/interim/uniprot_table_merged.txt'), 'w')

    for file in [insulin_edited, not_insulin_edited]:
        data.write(file.read())

    data.close()
    insulin_edited.close()
    not_insulin_edited.close()

    # Delete intermediate files
    subprocess.run(['rm',
        os.path.join(repo_dir, 'data/interim/insulin_table.txt'),
        os.path.join(repo_dir, 'data/interim/not_insulin_table.txt')])