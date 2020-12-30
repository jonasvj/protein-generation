#!/usr/bin/env python3
import os
import subprocess
from itertools import product
from src.utils import get_repo_dir

if __name__ == '__main__':

    repo_dir = get_repo_dir()

    data_file = open(
        os.path.join(repo_dir, 'data/interim/uniprot_table_merged.txt'), 'r')
    
    keyword_file = open(
        os.path.join(repo_dir, 'data/raw/uniprot_keywords.txt'), 'r')
    
    tidy_data_file = open(
        os.path.join(repo_dir, 'data/interim/uniprot_table_tidy.txt'), 'w')
    
    # Dictionary mapping keywords to keyword categories
    kw_cat_dict = dict()

    # Parse keyword file
    keyword_header = keyword_file.readline()
    for line in keyword_file:
        line = line.strip().split('\t')
        kw_name = line[1]
        kw_category = line[3]
        kw_cat_dict[kw_name] = kw_category

    keyword_file.close()

    # Keyword categories
    kw_cats = set(kw_cat_dict.values())
    kw_cats = sorted(kw_cats)

    # Use only biological process, cellular component and molecular function
    kw_cat_filter = ['biological', 'cellular', 'molecular']
    kw_cats_filtered = [kw for kw in kw_cats if
                        any([cat in kw.lower() for cat in kw_cat_filter])]
    
    # Header of data table
    data_header = data_file.readline().strip().split('\t')

    # Create new header 
    new_header = (data_header[0:1] + data_header[3:4] +
                  kw_cats_filtered + data_header[5:])
    tidy_data_file.write('\t'.join(new_header) + '\n')
    
    for line in data_file:
        id_, name, prot, organism, kws, pfams, seq, insulin = line.strip().split('\t')
        kws = kws.strip(';').split(';')
        
        # Skip sequences with no keywords
        if kws == ['']:
            continue
        
        # Arrange keywords by categories
        kws_by_cat = {cat: [] for cat in kw_cats_filtered}
        for kw in kws:
            kw_cat = kw_cat_dict[kw]

            # Keep only keywords in wanted categories
            if kw_cat in kw_cats_filtered:
                kws_by_cat[kw_cat].append(kw)
        
        # Add token for missing keyword
        n_kw_missing = 0
        for cat, kws in kws_by_cat.items():
            if len(kws) == 0:
                n_kw_missing += 1
                kws_by_cat[cat].append('None')
        
        # Skip if all keywords are missing 
        if n_kw_missing == len(kw_cats_filtered):
            continue
        
        new_row_values = [dict(zip(kws_by_cat, values))
                          for values in product(*kws_by_cat.values())]
        
        for row_values in new_row_values:
            new_row = ('{}\t'*len(new_header)).format(
                id_,
                organism,
                *[row_values[column] for column in kw_cats_filtered],
                pfams,
                seq,
                insulin)
            
            new_row = new_row.strip('\t') + '\n'
            tidy_data_file.write(new_row)
    
    data_file.close()
    tidy_data_file.close()