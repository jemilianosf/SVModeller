#!/usr/bin/env python3
# SVModeller - Shared Functions Module

import sys
import os

# Add script directory and parent root directory to sys.path for GAPI resolution
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if script_dir not in sys.path:
    sys.path.insert(0, script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import glob
import gzip
import random
import datetime
import subprocess
import warnings
import numpy as np
import pandas as pd
import pysam
import distfit
import os
from distfit import distfit
from GAPI import formats, gRanges

DEFAULT_INSERTION_EVENT_COLUMNS = [
    'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA', 'Alu__FOR+POLYA', 'Alu__TRUN+FOR+POLYA',
    'SVA__Alu-like+VNTR+SINE-R+POLYA', 'SVA__MAST2+VNTR+SINE-R+POLYA', 'L1__TRUN+FOR+POLYA+TD+POLYA',
    'L1__FOR+POLYA', 'SVA__VNTR+SINE-R+POLYA', 'L1__TRUN+FOR+POLYA', 'SVA__VNTR+SINE-R+POLYA+TD+POLYA',
    'SVA__SINE-R+POLYA', 'SVA__TD+MAST2+VNTR+SINE-R+POLYA', 'orphan', 'L1__TRUN+REV+DEL+FOR+POLYA',
    'SVA__TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA', 'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA',
    'L1__TD+FOR+POLYA', 'L1__TRUN+REV+DUP+FOR+POLYA', 'SVA__Alu-like+VNTR+SINE-R+POLYA+TD+POLYA',
    'L1__FOR+POLYA+TD+POLYA', 'L1__TRUN+REV+DUP+FOR+POLYA+TD+POLYA', 'L1__TRUN+REV+DEL+FOR+POLYA+TD+POLYA',
    'L1__TRUN+REV+BLUNT+FOR+POLYA', 'SVA__SINE-R+POLYA+TD+POLYA', 'SVA__MAST2+VNTR+SINE-R+POLYA+TD+POLYA',
    'L1__REV+DEL+FOR+POLYA', 'L1__TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA', 'VNTR', 'DUP', 'INV_DUP', 'NUMT'
]

DEFAULT_DELETION_EVENT_COLUMNS = [
    'simple', 'Alu__TRUN+FOR+POLYA', 'orphan', 'L1__TRUN+FOR+POLYA', 'SVA__TD+MAST2+VNTR+SINE-R+POLYA',
    'SVA__MAST2+VNTR+SINE-R+POLYA', 'L1__FOR+POLYA', 'L1__TD+FOR+POLYA', 'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA',
    'L1__TRUN+REV+DEL+FOR+POLYA', 'Alu__FOR+POLYA', 'SVA__VNTR+SINE-R+POLYA', 'L1__TRUN+REV+DEL+FOR+POLYA+TD+POLYA',
    'SVA__Alu-like+VNTR+SINE-R+POLYA', 'SVA__SINE-R+POLYA', 'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA',
    'L1__TRUN+FOR+POLYA+TD+POLYA', 'SVA__MAST2+VNTR+SINE-R+POLYA+TD+POLYA', 'L1__TRUN+REV+DUP+FOR+POLYA', 'VNTR'
]

def set_seed(seed: int = 42):
    """Sets random seeds across all libraries used in SVModeller

    to ensure reproducible results.
    """
    if seed is None:
        return

    # Set standard Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set Python hash seed for consistent dictionary/set ordering
    os.environ['PYTHONHASHSEED'] = str(seed)

def TD_filter(df):
    def check_conditions(row):
        is_5_valid = not (pd.isna(row['TD_5_Num']) or row['TD_5_Num'] == 'NA' or row['TD_5_Num'] == 0)
        is_3_valid = not (pd.isna(row['TD_3_Num']) or row['TD_3_Num'] == 'NA' or row['TD_3_Num'] == 0)
        if is_5_valid and is_3_valid:
            return 'BOTH'
        elif is_5_valid:
            return '5PRIME'
        elif is_3_valid:
            return '3PRIME'
        else:
            return 'NONE'
    df['TD_Status'] = df.apply(check_conditions, axis=1)
    df = df[df['TD_Status'] != 'BOTH']
    df = df.drop(columns=['TD_Status'])
    return df

def safe_int_val(val, default=0):
    if pd.isna(val) or val is None:
        return default
    s = str(val).strip().strip("'\"")
    if s.upper() in ('NA', 'NONE', 'NAN', '', 'N/A', '<NA>'):
        return default
    try:
        return int(float(s))
    except (ValueError, TypeError):
        return default

def extract_conformation_data(row):
    for_val, trun_val, rev_val, del_val, dup_val = (np.nan, np.nan, np.nan, np.nan, np.nan)
    conformation = str(row['Conformation'])
    conformation_ext = str(row['Conformation_Ext'])
    if 'FOR' in conformation:
        for_val = 1
    if 'TRUN' in conformation:
        trun_val = 1
    if 'REV' in conformation:
        rev_val = 1
    if 'DEL' in conformation:
        del_val = 1
    if 'DUP' in conformation:
        dup_val = 1
    if 'TRUN' in conformation_ext:
        trun_val = 1
    if 'REV' in conformation_ext:
        rev_val = 1
    if 'DEL' in conformation_ext:
        del_val = 1
    if 'DUP' in conformation_ext:
        dup_val = 1
    return pd.Series([for_val, trun_val, rev_val, del_val, dup_val])

def normalize_columns(df):
    event_columns = df.columns[3:]
    sum_per_event = df[event_columns].sum()
    sum_per_event.replace(0, 1, inplace=True)
    df[event_columns] = df[event_columns].div(sum_per_event, axis=1)
    return df

def sort_chromosomes(data):
    def key_func(chrom):
        if chrom.startswith('chr'):
            chrom = chrom[3:]
        if chrom.isdigit():
            return (0, int(chrom))
        elif chrom == 'X':
            return (1, 0)
        elif chrom == 'Y':
            return (2, 0)
        elif chrom == 'M':
            return (3, 0)
        else:
            return (4, chrom)

    if isinstance(data, pd.DataFrame):
        data['sort_key'] = data['#ref'].apply(key_func)
        sorted_df = data.sort_values('sort_key').drop(columns=['sort_key'])
        return sorted_df
    elif isinstance(data, list):
        return sorted(data, key=key_func)
    else:
        raise ValueError('Input must be a pandas DataFrame or a list of chromosome names.')

def filter_sd(dict_mutations, key_list):
    """
    Remove values outside +-2 standard deviations from the mean
    Only applied to numeric values
    """
    for key in key_list:
        if key in dict_mutations:
            data = np.array(dict_mutations[key])
            numeric_data = data[np.issubdtype(data.dtype, np.number)]
            if len(numeric_data) > 0:
                mean = np.mean(numeric_data)
                std = np.std(numeric_data)
                filtered_data = [x for x in dict_mutations[key] if isinstance(x, (int, float)) and (mean - 2 * std <= x <= mean + 2 * std)]
                dict_mutations[key] = filtered_data
    return dict_mutations

def read_vcf_file_BED(file_path, sv_type='insertion'):
    vcf = formats.VCF()
    vcf.read(file_path)
    data = []
    type_key = 'ITYPE_N' if sv_type == 'insertion' else 'DTYPE_N'
    for variant in vcf.variants:
        canonical_status = 'NOT_CANONICAL' if 'NOT_CANONICAL' in variant.info else ''
        length = len(variant.alt) if sv_type == 'insertion' else variant.info.get('DEL_LEN', 'NA')
        row_data = {
            'Type_SV': variant.info.get(type_key, 'NA'),
            'Family': variant.info.get('FAM_N', 'NA'),
            'Conformation': variant.info.get('CONFORMATION', 'NA'),
            'Conformation_Ext': variant.info.get('CONFORMATION_EXT', 'NA'),
            'Start_position': variant.pos,
            'End_position': variant.pos + len(variant.alt),
            'Length': length,
            'Chromosome': variant.chrom,
            'PolyA_Length': variant.info.get('POLYA_LEN', 'NA'),
            'Strand': variant.info.get('STRAND', 'NA'),
            'TSD_Length': variant.info.get('TSD_LEN', 'NA'),
            'TD_5_Num': variant.info.get('5PRIME_NB_TD', 'NA'),
            'TD_5': variant.info.get('5PRIME_TD_LEN', 'NA'),
            'TD_3_Num': variant.info.get('3PRIME_NB_TD', 'NA'),
            'TD_3': variant.info.get('3PRIME_TD_LEN', 'NA'),
            'SVA_Hexamer': variant.info.get('HEXAMER_LEN', 'NA'),
            'TD_orphan_Length': variant.info.get('ORPHAN_TD_LEN', 'NA'),
            'VNTR_Num_Motifs': variant.info.get('NB_MOTIFS', 'NA'),
            'VNTR_Motifs': variant.info.get('MOTIFS', 'NA'),
            'Canonical': canonical_status,
            'SVA_VNTR_Length': variant.info.get('VNTR_LEN', 'NA'),
            'SVA_VNTR_Coordinates': variant.info.get('VNTR_COORD', 'NA')
        }
        if sv_type == 'insertion':
            row_data['Complete_Sequence'] = variant.alt
        data.append(row_data)

    columns = ['Chromosome', 'Start_position', 'End_position', 'Type_SV', 'Family', 'Conformation', 'Conformation_Ext', 'Canonical', 'Length', 'PolyA_Length', 'Strand', 'TSD_Length', 'TD_5_Num', 'TD_5', 'TD_3_Num', 'TD_3', 'SVA_Hexamer', 'SVA_VNTR_Length', 'TD_orphan_Length', 'VNTR_Num_Motifs', 'VNTR_Motifs', 'SVA_VNTR_Coordinates']
    if sv_type == 'insertion':
        columns.append('Complete_Sequence')
    df = pd.DataFrame(data, columns=columns)
    return df

def process_bed_table(result_df, sv_type='insertion'):
    result_BED_table = result_df.rename(columns={'Type_SV': 'name', 'Chromosome': '#ref', 'Start_position': 'beg', 'End_position': 'end', 'Family': 'SubType'})
    if sv_type == 'insertion':
        result_BED_table = result_BED_table[result_BED_table['name'] != 'NA']
    else:
        result_BED_table['name'] = result_BED_table['name'].replace('NA', 'simple')

    result_BED_table = result_BED_table[~((result_BED_table['Canonical'] == 'NOT_CANONICAL') & (result_BED_table['name'] != 'orphan'))]
    result_BED_table.drop('Canonical', axis=1, inplace=True)
    result_BED_table = result_BED_table[result_BED_table['name'] != 'DUP_INTERSPERSED']
    result_BED_table = result_BED_table[result_BED_table['name'] != 'COMPLEX_DUP']

    vntr_df = None
    if sv_type == 'insertion':
        vntr_df = extract_vntr_with_start(result_BED_table)
        result_BED_table = SVA_VNTR_Motif(result_BED_table)
        result_BED_table = extract_SVA_VNTR_Motifs(result_BED_table)

    result_BED_table['name'] = result_BED_table.apply(lambda row: row['SubType'] if row['name'] in ['solo', 'partnered'] else row['name'], axis=1)
    result_BED_table = result_BED_table.drop('SubType', axis=1)
    result_BED_table = TD_filter(result_BED_table)

    columns_to_convert = ['Length', 'PolyA_Length', 'TSD_Length', 'TD_5_Num', 'TD_5', 'TD_3_Num', 'TD_3', 'SVA_Hexamer', 'SVA_VNTR_Length', 'TD_orphan_Length', 'VNTR_Num_Motifs']
    for col in columns_to_convert:
        result_BED_table[col] = pd.to_numeric(result_BED_table[col], errors='coerce')
    result_BED_table['SVA_Hexamer'] = pd.to_numeric(result_BED_table['SVA_Hexamer'], errors='coerce')
    for col in result_BED_table.columns:
        result_BED_table[col] = result_BED_table[col].astype(object).fillna('NA')
    result_BED_table[['FOR', 'TRUN', 'REV', 'DEL', 'DUP']] = result_BED_table.apply(extract_conformation_data, axis=1)

    if sv_type == 'deletion':
        result_BED_table['Event'] = result_BED_table.apply(lambda row: f"{row['name']}__{row['Conformation']}", axis=1)
        result_BED_table['Event'] = result_BED_table['Event'].replace({'VNTR__NA': 'VNTR', 'DUP__NA': 'DUP', 'INV_DUP__NA': 'INV_DUP', 'NUMT__NA': 'NUMT', 'orphan__NA': 'orphan', 'simple__NA': 'simple'})
        result_BED_table = result_BED_table.drop('Conformation_Ext', axis=1)
        return result_BED_table

    return (result_BED_table, vntr_df)

def probabilities_df(table, output_path=None):
    name_distribution_df = table['Event'].value_counts().reset_index()
    name_distribution_df.columns = ['Event', 'number']
    total = name_distribution_df['number'].sum()
    name_distribution_df['Probability'] = name_distribution_df['number'] / total
    name_distribution_df = name_distribution_df.drop(columns=['number'])
    if output_path:
        name_distribution_df.to_csv(output_path, sep='	', index=False)
    return name_distribution_df

def mut_bins(bins, table, event_columns=None):
    if event_columns is None:
        event_columns = DEFAULT_INSERTION_EVENT_COLUMNS
    df = pd.DataFrame(columns=['window', 'beg', 'end'] + event_columns)
    for window in bins:
        chrom, start, end = window
        subset = table[(table['#ref'] == chrom) & (table['beg'] >= start) & (table['end'] <= end)]
        mutation_counts = subset['Event'].value_counts().to_dict()
        row = {'window': f"{chrom}:{start}-{end}", 'beg': start, 'end': end}
        for mutation in event_columns:
            row[mutation] = mutation_counts.get(mutation, 0)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    return df

def SVA_VNTR_Motif(df):
    # Function to extract motif based on coordinates
    def extract_motif(row):
        coordinates = row['SVA_VNTR_Coordinates']
        sequence = row['Complete_Sequence']

        if pd.notna(coordinates) and '-' in coordinates:
            start, end = map(int, coordinates.split('-'))
            # Extract the segment from the sequence using the coordinates
            return sequence[start:end]
        return None  # Return None if there are no coordinates or the format is incorrect

    # Apply the extract_motif function to each row
    df['SVA_VNTR_Motif'] = df.apply(extract_motif, axis=1)

    # Drop the 'SVA_VNTR_Coordinates' and 'Complete_Sequence' columns
    df = df.drop(columns=['SVA_VNTR_Coordinates', 'Complete_Sequence'])

    return df

def extract_SVA_VNTR_Motifs(df):
    filename="SVA_VNTR_Motifs.txt"
    with open(filename, "w") as f:
        # Iterate through each row in the dataframe
        for motif in df['SVA_VNTR_Motif']:
            # Only write non-null motifs to the file
            if pd.notna(motif):
                f.write(str(motif) + "\n")

    # Drop the 'SVA_VNTR_Motif' column from the DataFrame
    df = df.drop(columns=['SVA_VNTR_Motif'])

    return df

def extract_vntr_with_start(df):
    # Extract the from the VNTRs the start position, complete sequence, number of motifs, and the motifs
    df_clean = df[['beg', 'Complete_Sequence', 'VNTR_Num_Motifs', 'VNTR_Motifs']].copy()
    df_clean.rename(columns={'beg': 'Start'}, inplace=True)

    # rRemove any case with NA
    mask = ~(df_clean.astype(str).apply(lambda x: x.str.strip().str.upper()).eq("NA").any(axis=1))
    df_clean = df_clean[mask]

    # Save
    df_clean.to_csv("VNTR_with_start_position.txt", sep='\t', index=False)

    return df_clean

def create_dict(df):
    # Create the 'Event' column: Combine the column name and 'Conformation'
    df['Event'] = df.apply(lambda row: f"{row['name']}__{row['Conformation']}", axis=1)
    df['Event'] = df['Event'].replace(
            {'VNTR__NA': 'VNTR','DUP__NA': 'DUP', 'INV_DUP__NA': 'INV_DUP', 'NUMT__NA': 'NUMT', 'orphan__NA': 'orphan'}
        )
    # Create an empty dictionary to store results
    event_dict = {}

    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        event = row['Event']  # Get the event value for the current row

        # Iterate over the columns from 'Length' (column 7) to the end
        for col in df.columns[6:]:  # Column 7 is index 7 (starting from 'Strand')
            value = row[col]  # Get the value in the current column
            if value != 'NA' and value != 'NA' and value is not None:  # Check if the value is not 'NA' or None
                # Construct the dictionary key
                key = f"{event}__{col}"

                # Add the value to the dictionary (create a list if key does not exist)
                if key not in event_dict:
                    event_dict[key] = []
                event_dict[key].append(value)

    # Remove keys that end with 'Event'
    keys_to_remove = [key for key in event_dict if key.endswith('Event')]

    # Delete the keys from the dictionary
    for key in keys_to_remove:
        del event_dict[key]

    return event_dict

def process_dictionary(dict_mutations):
    '''
    Process the dictionary by applying the filter_sd function for specific keys
    and extracting VNTR motifs, then saving the motifs as a TSV file.

    Arguments:
    - dict_mutations (dict): The dictionary to process.

    Returns:
    - dict_mutations (dict): The modified dictionary after applying both functions.
    '''
    # Step 1: Apply filter_sd to all keys in the dictionary
    keys_filter = list(dict_mutations.keys())
    dict_mutations = filter_sd(dict_mutations.copy(), keys_filter)

    # Return the modified dictionary and the DataFrame containing the motifs and their proportions
    return dict_mutations

def insertion_features_df(input_dict):
    # Define the columns as per the requirement
    columns = [
        'Event', 'Length', 'Strand', 'TSD_Length', 'TD_5', 'TD_3', 'SVA_Hexamer',
        'SVA_VNTR_Length', 'TD_orphan_Length', 'VNTR_Num_Motifs',
        'PolyA_Length_1', 'PolyA_Length_2', 'FOR', 'TRUN', 'REV', 'DEL', 'DUP'
    ]

    # Define the possible events that correspond to rows in the DataFrame
    event_column = [
        'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA', 'Alu__FOR+POLYA',
        'Alu__TRUN+FOR+POLYA', 'SVA__Alu-like+VNTR+SINE-R+POLYA',
        'SVA__MAST2+VNTR+SINE-R+POLYA', 'L1__TRUN+FOR+POLYA+TD+POLYA',
        'L1__FOR+POLYA', 'SVA__VNTR+SINE-R+POLYA', 'L1__TRUN+FOR+POLYA',
        'SVA__VNTR+SINE-R+POLYA+TD+POLYA', 'SVA__SINE-R+POLYA',
        'SVA__TD+MAST2+VNTR+SINE-R+POLYA', 'orphan',
        'L1__TRUN+REV+DEL+FOR+POLYA', 'SVA__TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA',
        'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA',
        'L1__TD+FOR+POLYA', 'L1__TRUN+REV+DUP+FOR+POLYA',
        'SVA__Alu-like+VNTR+SINE-R+POLYA+TD+POLYA', 'L1__FOR+POLYA+TD+POLYA',
        'L1__TRUN+REV+DUP+FOR+POLYA+TD+POLYA',
        'L1__TRUN+REV+DEL+FOR+POLYA+TD+POLYA',
        'L1__TRUN+REV+BLUNT+FOR+POLYA', 'SVA__SINE-R+POLYA+TD+POLYA',
        'SVA__MAST2+VNTR+SINE-R+POLYA+TD+POLYA', 'L1__REV+DEL+FOR+POLYA',
        'L1__TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA', 'VNTR', 'DUP',
        'INV_DUP', 'NUMT'
    ]

    # Step 1: Process the dictionary values to convert floats to integers
    for key, value in input_dict.items():
        # Convert floats to integers, leaving other types unchanged
        input_dict[key] = [
            (int(x) if isinstance(x, float) else x) for x in value
        ]

    # Create an empty DataFrame with the specified columns
    df = pd.DataFrame(columns=columns)

    # Step 2: Process the dictionary and add data to the DataFrame
    for key, values in input_dict.items():
        # Split the key into event name and column name (e.g., 'SVA_Hexamer+Alu-like+VNTR+SINE-R+POLYA_Length')
        event_name, column_name = key.rsplit('__', 1)

        # Check if the event name is valid (exists in the event_column list)
        if event_name in event_column and column_name in columns:
            # If event_name is not already in the DataFrame, create a new row with empty values
            if event_name not in df['Event'].values:
                # Create an empty row and add the event name
                empty_row = {col: '' for col in columns}
                empty_row['Event'] = event_name
                df = pd.concat([df, pd.DataFrame([empty_row])], ignore_index=True)

            # Get the index of the row that corresponds to the current event
            event_index = df[df['Event'] == event_name].index[0]

            # Assign the values to the correct cell, join by commas in case of multiple values
            # Avoid overwriting cells if they already have values
            existing_value = df.at[event_index, column_name]
            if existing_value:
                new_value = ','.join(map(str, values)) if values else ''
                # Check for duplicates and join without repeating values
                existing_value = set(existing_value.split(','))
                new_value = set(new_value.split(','))
                final_value = list(existing_value | new_value)
                df.at[event_index, column_name] = ','.join(final_value)
            else:
                df.at[event_index, column_name] = ','.join(map(str, values)) if values else ''

    # Save the Insertion Features df to a .tsv file
    df.to_csv('Insertion_Features.tsv', sep='\t', index=False)

def genome_wide_distribution(chromosome_length, bin_size, table):
    # Dictionary containing references as keys and their lengths as values:
    chr_length = formats.chrom_lengths_index(chromosome_length)

    # Bin size
    binSize = bin_size  # this is 1MB

    # Use chromosomes present in chromosome_length file
    chromosomes = list(chr_length.keys())

    # Create genomic bins based on chromosome lengths and bin size
    bins = gRanges.makeGenomicBins(chr_length, binSize, chromosomes)[::-1]

    # Create the table of insertions classified in windows
    res_table = mut_bins(bins, table)
    final_table = normalize_columns(res_table)

    # Save the result to a TSV file
    final_table.to_csv('Genome_Wide_Distribution.tsv', sep='\t', index=False)

def consensus_seqs(file_path):
    '''
    Function to read from a fasta file the consensus sequences

    Input: path of the fasta file
    Output: dictionary containing, for example, key: Alu_Seq - Value: the consensus sequence
    '''
    sequences = {"Alu_Seq": "", "L1_Seq": "", "SVA_Alu-like_Seq": "", "SVA_SINE-R_Seq": "",
                "SVA_MAST2_Seq": "", "NUMT_Seq": ""}

    _open = gzip.open if str(file_path).endswith('.gz') else open
    with _open(file_path, "rt") as file:
        lines = file.readlines()

    current_sequence = ""
    previous_header = ""  # Initialize previous_header here to avoid the UnboundLocalError

    for i in range(len(lines)):
        line = lines[i].strip()  # Remove any leading/trailing spaces or newlines

        if line.startswith(">"):
            # If we were already collecting a sequence, assign it to the corresponding key
            if current_sequence:
                if "consensus|Alu" in previous_header:
                    sequences["Alu_Seq"] = current_sequence
                elif "consensus|L1" in previous_header:
                    sequences["L1_Seq"] = current_sequence
                elif "consensus|SVA|SVA_F|Alu-like" in previous_header:
                    sequences["SVA_Alu-like_Seq"] = current_sequence
                elif "consensus|SVA|SVA_F|SINE-R" in previous_header:
                    sequences["SVA_SINE-R_Seq"] = current_sequence
                elif "consensus|SVA|exon1|MAST2" in previous_header:
                    sequences["SVA_MAST2_Seq"] = current_sequence
                elif "consensus|NC_012920.1" in previous_header:
                    sequences["NUMT_Seq"] = current_sequence
            # Reset current sequence and store the new header
            current_sequence = ""
            previous_header = line  # Keep track of the header to check for matching keys
        else:
            # Append the current line (sequence part) to the current sequence
            current_sequence += line.strip()  # Remove extra spaces or newlines

    # Don't forget to handle the last sequence
    if current_sequence:
        if "consensus|Alu" in previous_header:
            sequences["Alu_Seq"] = current_sequence
        elif "consensus|L1" in previous_header:
            sequences["L1_Seq"] = current_sequence
        elif "consensus|SVA|SVA_F|Alu-like" in previous_header:
            sequences["SVA_Alu-like_Seq"] = current_sequence
        elif "consensus|SVA|SVA_F|SINE-R" in previous_header:
            sequences["SVA_SINE-R_Seq"] = current_sequence
        elif "consensus|SVA|exon1|MAST2" in previous_header:
            sequences["SVA_MAST2_Seq"] = current_sequence
        elif "consensus|NC_012920.1" in previous_header:
            sequences["NUMT_Seq"] = current_sequence

    return sequences

def probabilities_total_number(probabilities_numbers_df,num_events):
    table = pd.read_csv(probabilities_numbers_df, sep='\t', compression='infer')
    print(f"Columns in the table: {table.columns}")
    if 'Probability' in table.columns:
        sampled_names = np.random.choice(table['Event'], size=num_events, p=table['Probability'])
        table_events = pd.DataFrame({'name': sampled_names})
    elif 'Number' in table.columns:
        rows = []
        for i, row in table.iterrows():
            event = row['Event']
            count = row['Number']
            rows.extend([event] * int(count))
        table_events = pd.DataFrame({'name': rows})
    else:
        raise ValueError("The second column must be either 'Probability' or 'Number'.")

    return table_events

def generate_dict_from_table(data):
    result_dict = {}

    for index, row in data.iterrows():
        event = row['Event']

        for column in data.columns[1:]:  # Skip the 'Event' column
            value = row[column]

            if pd.notna(value):  # Check if value is not NaN
                key = f"{event}__{column}"
                str_val = str(value)
                result_dict[key] = [int(float(x)) for x in str_val.split(',') if x.strip().replace('-', '').replace('.', '').isdigit()]

    # Remove keys that end with 'Strand' and 'Length'
    result_dict = {key: value for key, value in result_dict.items() if not key.endswith('Strand')}

    return result_dict

def generate_random_numbers(dictionary, num_samples):
    random_numbers_dict = {}
    for key, value in dictionary.items():
        dist = distfit()
        dist.fit_transform(value, verbose = 0)
        random_numbers = dist.generate(num_samples)
        random_numbers = random_numbers.astype(int)
        random_numbers_dict[key] = random_numbers
    return random_numbers_dict

def remove_negative_values(values):
    '''
    Function to remove negative values from the list
    '''
    return [value for value in values if value > 0]

def filter_DEL(dict):
    '''
    Check if a key contains 'DEL' in its name. If it does, it removes values larger than 1000 from the corresponding list of values for that key.
    '''
    for key in dict:
        if 'DEL' in key:
            # Get the list of values for this key
            values = dict[key]
            # Filter out values greater than 1000
            filtered_values = [value for value in values if value <= 1000]
            # Update the dictionary with the filtered values
            dict[key] = filtered_values
    return dict

def filter_FOR(dict_insertion_features, dict_consensus):
    # Extract the Alu_Seq and L1_Seq from dict_consensus and get their lengths
    aluseq_seq = dict_consensus.get('Alu_Seq', '')
    l1seq_seq = dict_consensus.get('L1_Seq', '')

    aluseq_length = len(aluseq_seq) if aluseq_seq else 0
    l1seq_length = len(l1seq_seq) if l1seq_seq else 0

    # Iterate over the dictionary
    for key, values in dict_insertion_features.items():
        if 'FOR' in key and key.endswith('__FOR'):  # Check if 'FOR' is in key and ends with '__FOR'
            if 'L1' in key:
                # Remove values larger than L1_seq length for 'FOR' and 'L1' keys
                dict_insertion_features[key] = [value for value in values if value <= l1seq_length]
            elif 'Alu' in key:
                # Remove values larger than Alu_seq length for 'FOR' and 'Alu' keys
                dict_insertion_features[key] = [value for value in values if value <= aluseq_length]

    return dict_insertion_features

def distribution_random_numbers(dict_insertion_features,num_events,dict_consensus):
    array_dict = {}

    for key in dict_insertion_features:
        array = np.array(dict_insertion_features[key])
        array_dict[key] = array

    number_random_events = num_events * 3
    random_numbers_dict = generate_random_numbers(array_dict, number_random_events)

    # Remove negative values
    for key, values in random_numbers_dict.items():
        random_numbers_dict[key] = remove_negative_values(values)

    # Filter values outside ±2 standard deviations
    keys_filter = list(random_numbers_dict.keys())
    random_numbers_dict = filter_sd(random_numbers_dict, keys_filter)

    # Filter largest DEL values
    random_numbers_dict = filter_DEL(random_numbers_dict)

    # Filter FOR values larger than consensus sequences
    random_numbers_dict = filter_FOR(random_numbers_dict, dict_consensus)

    return random_numbers_dict

def process_insertion_features_random_numbers(insertion_features_df,num_events,dict_consensus):
    # Open insertion features df
    table_insertion_features = pd.read_csv(insertion_features_df, sep='\t', compression='infer')
    # Create dictionary from the df
    dict_insertion_features = generate_dict_from_table(table_insertion_features)
    # Generate distributions of data (disfit) and dictionary of random numbers for every event and feature
    dict_random = distribution_random_numbers(dict_insertion_features,num_events,dict_consensus)
    return dict_random

def add_beg_end_columns(df_insertions, genome_wide_distribution_df):
    '''
    Function to add ref and beg columns to the df of insertions based on genome-wide distribution
    '''
    # Open insertion features df
    genome_wide_distribution = pd.read_csv(genome_wide_distribution_df, sep='\t', compression='infer')

    # Create new columns in the first DataFrame
    df_insertions['#ref'] = None
    df_insertions['beg'] = None

    # Function to select a random row based on probabilities
    def select_random_row(probabilities):
        prob_vals = pd.to_numeric(probabilities, errors='coerce').fillna(0).values.astype(float)
        total_prob = np.sum(prob_vals)
        if total_prob > 0:
            norm_probs = prob_vals / total_prob
            return np.random.choice(probabilities.index, p=norm_probs)
        else:
            return np.random.choice(probabilities.index)

    # Iterate over each row in the first DataFrame
    for index, row in df_insertions.iterrows():
        event_name = str(row['name'])
        base_event = event_name.split('__')[0]

        if event_name in genome_wide_distribution.columns:
            probabilities = genome_wide_distribution[event_name]
        elif base_event in genome_wide_distribution.columns:
            probabilities = genome_wide_distribution[base_event]
        else:
            probabilities = pd.Series(1.0, index=genome_wide_distribution.index)

        selected_row = select_random_row(probabilities)

        # Fill the values in the first DataFrame
        df_insertions.at[index, '#ref'] = str(genome_wide_distribution.at[selected_row, 'window']).split(':')[0]
        df_insertions.at[index, 'beg'] = np.random.randint(genome_wide_distribution.at[selected_row, 'beg'], genome_wide_distribution.at[selected_row, 'end'])

    return df_insertions

def add_elements_columns(dict, df_insertions):

    # Create new columns with default value 0
    columns = ['Length', 'PolyA_Length_1', 'PolyA_Length_2', 'TSD_Length', 'TD_5', 'TD_3', 'TD_orphan_Length',
               'VNTR_Num_Motifs', 'SVA_Hexamer', 'SVA_VNTR_Length', 'FOR', 'TRUN', 'REV', 'DEL', 'DUP']

    for column in columns:
        df_insertions[column] = 0

    # Iterate over rows in df_insertions
    for index, row in df_insertions.iterrows():
        name = row['name']
        for column in columns:
            key = f"{name}__{column}"
            # Check if the key exists in the dictionary and has corresponding values
            if key in dict and dict[key]:
                # Select a random value from the list without modifying the dictionary
                df_insertions.loc[index, column] = random.choice(dict[key])
            else:
                # If the key is not found or the list is empty, assign 0
                df_insertions.loc[index, column] = 0

    # Fill any NaN values with 0
    df_insertions.fillna(0, inplace=True)

    return df_insertions

def add_SVA_info(df, dict_consensus):
    # Extract the sequence data from dict_consensus
    aluseq_seq = dict_consensus.get('SVA_Alu-like_Seq', '')
    siner_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    mast2_seq = dict_consensus.get('SVA_MAST2_Seq', '')

    # Get the lengths of the sequences
    aluseq_length = len(aluseq_seq) if aluseq_seq else 0
    siner_length = len(siner_seq) if siner_seq else 0
    mast2_length = len(mast2_seq) if mast2_seq else 0

    # Create the new columns in the dataframe
    df['SINE_R'] = 0
    df['MAST2'] = 0
    df['ALU_LIKE'] = 0

    # Iterate through each row and check the 'name' column
    for index, row in df.iterrows():
        if 'Alu-like' in row['name']:
            df.at[index, 'ALU_LIKE'] = aluseq_length
        if 'SINE-R' in row['name']:
            df.at[index, 'SINE_R'] = siner_length
        if 'MAST2' in row['name']:
            df.at[index, 'MAST2'] = mast2_length

    return df

# Function to calculate the Length for each row
def calculate_length(row, columns_to_sum):
    # Check if 'name' is one of the excluded values
    if row['name'] in ['NUMT', 'DUP', 'INV_DUP', 'VNTR']:
        return row['Length']  # Do not change the Length for these rows

    # Sum all the relevant columns, considering missing values as 0
    total_sum = 0
    for col in columns_to_sum:
        # Ensure the column value is numeric, convert if necessary
        value = pd.to_numeric(row.get(col, 0), errors='coerce')  # Convert to numeric, coercing errors to NaN
        total_sum += value if not pd.isna(value) else 0  # Add the value or 0 if NaN

    # Subtract the value in the 'DEL' column, if it exists
    del_value = pd.to_numeric(row.get('DEL', 0), errors='coerce')
    total_sum -= del_value if not pd.isna(del_value) else 0  # Subtract the value or 0 if NaN

    return total_sum

def update_trun(df_inertions, dict_consensus):
    # Extract the Alu_Seq and L1_Seq from dict_consensus and get their lengths
    aluseq_seq = dict_consensus.get('Alu_Seq', '')
    l1seq_seq = dict_consensus.get('L1_Seq', '')

    # Get the length of the DNA sequences (length of the string)
    aluseq_length = len(aluseq_seq) if aluseq_seq else 0
    l1seq_length = len(l1seq_seq) if l1seq_seq else 0

    # Iterate over the DataFrame rows
    for index, row in df_inertions.iterrows():
        name_value = str(row['name'])

        # Calculate the sum of 'FOR', 'DEL', and 'REV' values, treating NaN as 0
        total_subtraction = 0
        for col in ['FOR', 'DEL', 'REV']:
            value = row.get(col, 0)

            # Ensure the value is numeric (if not, treat it as 0)
            try:
                value = float(value)
            except ValueError:
                value = 0  # If it's a non-numeric value, treat it as 0

            total_subtraction += value

        # Check if both 'L1' or 'Alu' and 'TRUN' are in the 'name' column and update 'TRUN' accordingly
        if 'L1' in name_value and 'TRUN' in name_value:
            # Calculate TRUN for 'L1'
            df_inertions.at[index, 'TRUN'] = l1seq_length - total_subtraction
        elif 'Alu' in name_value and 'TRUN' in name_value:
            # Calculate TRUN for 'Alu'
            df_inertions.at[index, 'TRUN'] = aluseq_length - total_subtraction

    return df_inertions

# Function to update the DataFrame
def update_dataframe(df_insertions, dict_consensus):
    # Add a column 'Strand' with randomly assigned '+' or '-'
    df_insertions['Strand'] = np.random.choice(['+', '-'], size=len(df_insertions))

    # List of columns to sum
    columns_to_sum = [
        'PolyA_Length_1', 'PolyA_Length_2', 'TSD_Length', 'TD_5', 'TD_3', 'TD_orphan_Length',
        'SVA_Hexamer', 'SVA_VNTR_Length', 'FOR', 'REV', 'DUP', 'SINE_R', 'MAST2', 'ALU_LIKE'
    ]

    # Apply the calculate_length function to each row and update the 'Length' column
    df_insertions['Length'] = df_insertions.apply(calculate_length, axis=1, columns_to_sum=columns_to_sum)

    # Update the 'TRUN' column based on dict_consensus (Alu_Seq and L1_Seq lengths)
    df_insertions = update_trun(df_insertions, dict_consensus)

    # Remove possible rows where the 'Length' column is negative
    df_insertions = df_insertions[df_insertions['Length'] >= 0]

    return df_insertions

def add_source_gene_info(df_insertions, source_L1_path, source_SVA_path):
    # Load the source element tables
    table_source_L1 = pd.read_csv(source_L1_path, sep='\t', compression='infer')
    table_source_SVA = pd.read_csv(source_SVA_path, sep='\t', compression='infer')

    # Add necessary columns with default values
    df_insertions[['SRC_identifier', 'SRC_ref', 'SRC_beg', 'SRC_end', 'SRC_cont_PCAWG', 'SRC_strand', 'SRC_in_ref_genome']] = 0

    # Iterate over rows to assign values from table_source_L1 and table_source_SVA
    for index, row in df_insertions.iterrows():
        # If 'L1' AND 'TD' in name, assign values from L1 table
        if 'L1' in row['name'] and 'TD' in row['name']:
            probabilities = table_source_L1['SRC_cont_PCAWG'].values
            selected_row = table_source_L1.sample(weights=probabilities).iloc[0]
            df_insertions.loc[index, ['SRC_identifier', 'SRC_ref', 'SRC_beg', 'SRC_end', 'SRC_cont_PCAWG', 'SRC_strand', 'SRC_in_ref_genome']] = selected_row.values
        # If 'orphan' AND 'TD' in name, assign values assuming orphan gets from L1
        elif row['name'] == 'orphan':
            probabilities = table_source_L1['SRC_cont_PCAWG'].values
            selected_row = table_source_L1.sample(weights=probabilities).iloc[0]
            df_insertions.loc[index, ['SRC_identifier', 'SRC_ref', 'SRC_beg', 'SRC_end', 'SRC_cont_PCAWG', 'SRC_strand', 'SRC_in_ref_genome']] = selected_row.values
        # If 'SVA' AND 'TD' in name, assign values from SVA table
        elif 'SVA' in row['name'] and 'TD' in row['name']:
            probabilities = table_source_SVA['SRC_contribution'].values
            selected_row = table_source_SVA.sample(weights=probabilities).iloc[0]
            df_insertions.loc[index, ['SRC_identifier', 'SRC_ref', 'SRC_beg', 'SRC_end', 'SRC_contribution', 'SRC_strand', 'SRC_in_ref_genome']] = selected_row.values

    # Fill NaN values with 0
    df_insertions.fillna(0, inplace=True)

    # Add new columns for transduction start and end
    df_insertions[['TD_beg', 'TD_end']] = 0

    # Modify the values of 'TD_beg' and 'TD_end' based on conditions
    for index, row in df_insertions.iterrows():
        if 'L1' in row['name'] and 'TD' in row['name']:
            if row['SRC_strand'] == 'plus':
                df_insertions.at[index, 'TD_beg'] = row['SRC_end']
                df_insertions.at[index, 'TD_end'] = row['SRC_end'] + row['TD_5'] if row['TD_5'] != 0 else row['SRC_end'] + row['TD_3']
            elif row['SRC_strand'] == 'minus':
                df_insertions.at[index, 'TD_end'] = row['SRC_beg']
                df_insertions.at[index, 'TD_beg'] = row['SRC_beg'] - row['TD_5'] if row['TD_5'] != 0 else row['SRC_beg'] - row['TD_3']
        elif 'SVA' in row['name'] and 'TD' in row['name']:
            if row['SRC_strand'] == 'plus':
                df_insertions.at[index, 'TD_beg'] = row['SRC_end']
                df_insertions.at[index, 'TD_end'] = row['SRC_end'] + row['TD_5'] if row['TD_5'] != 0 else row['SRC_end'] + row['TD_3']
            elif row['SRC_strand'] == 'minus':
                df_insertions.at[index, 'TD_end'] = row['SRC_beg']
                df_insertions.at[index, 'TD_beg'] = row['SRC_beg'] - row['TD_5'] if row['TD_5'] != 0 else row['SRC_beg'] - row['TD_3']
        elif 'orphan' in row['name']:
            if row['SRC_strand'] == 'plus':
                df_insertions.at[index, 'TD_beg'] = row['SRC_end']
                df_insertions.at[index, 'TD_end'] = row['SRC_end'] + row['TD_orphan_Length']
            elif row['SRC_strand'] == 'minus':
                df_insertions.at[index, 'TD_end'] = row['SRC_beg']
                df_insertions.at[index, 'TD_beg'] = row['SRC_beg'] - row['TD_orphan_Length']

    # Update SRC_strand values
    df_insertions['SRC_strand'] = df_insertions['SRC_strand'].replace({'minus': '-', 'plus': '+'})

    # Convert 0s to 'NA', except for 'SRC_in_ref_genome' column
    keep_columns = ['SRC_in_ref_genome']
    for column in df_insertions.columns:
        if column not in keep_columns:
            df_insertions[column] = df_insertions[column].apply(lambda x: 'NA' if x == 0 else x)

    # Directly update 'SRC_in_ref_genome' based on 'SRC_identifier'
    df_insertions['SRC_in_ref_genome'] = df_insertions.apply(lambda row: 'NA' if row['SRC_identifier'] == 'NA' else row['SRC_in_ref_genome'], axis=1)

    return df_insertions

# Open and get SVAs VNTR motifs
def read_file_and_store_lines(file_path):
    _open = gzip.open if str(file_path).endswith('.gz') else open
    with _open(file_path, 'rt') as file:
        lines = [line.strip() for line in file]
    return lines

def VNTR_insertions(row, motifs_file):
    '''
    Function to generate VNTR sequences by selecting a random row from the provided file
    and using its 'Complete_Sequence', 'Start', 'VNTR_Num_Motifs', and 'VNTR_Motifs'.
    '''

    # Step 1: Read the motifs file
    motifs_df = pd.read_csv(motifs_file, sep='\t', compression='infer')  # Reads the .tsv file containing motif information

    # Step 2: Randomly select a row from the motifs file
    random_row = motifs_df.sample(n=1).iloc[0]

    # Step 3: Get values from the randomly selected row
    complete_sequence = random_row['Complete_Sequence']
    start_position = random_row['Start']
    vntr_num_motifs = int(random_row['VNTR_Num_Motifs'])
    vntr_motifs = random_row['VNTR_Motifs'].split(',')
    # Step 4: Update the row with the selected values
    row['Sequence_Insertion'] = complete_sequence
    row['Length'] = len(complete_sequence)  # Set the Length as the length of the Complete_Sequence
    row['VNTR_Num_Motifs'] = vntr_num_motifs  # Update the VNTR_Num_Motifs
    row['VNTR_Motifs'] = vntr_motifs  # Update the VNTR_Motifs list
    row['Start'] = start_position  # Update the Start position

    # Step 5: Return the Complete_Sequence as the sequence and the VNTR_Motifs
    sequence = complete_sequence  # Now the sequence is just the Complete_Sequence
    return sequence, vntr_motifs, vntr_num_motifs

def DUP_insertions(row, reference_fasta):
    '''
    Function to generate the duplicated sequences
    '''
    start = safe_int_val(row['beg'])
    length = safe_int_val(row.get('Length', 100))
    if length <= 0:
        length = 100
    end = start + length
    with pysam.FastaFile(reference_fasta) as fasta_file:
        insertion = fasta_file.fetch(row['#ref'], start, max(start + 1, end))
    return insertion

def NUMT_insertions(row, dict_consensus):
    '''
    Function to generate the mitochondrial insertion sequences
    '''
    # Retrieve the sequence from the dictionary
    seq = dict_consensus.get('NUMT_Seq', '')
    if not seq:
        seq = "GATCACAGGTCTATCACCCTATTAACCACTCACGGGAGCTCTCCATGC"

    # Transform Total_Length to integer
    length = safe_int_val(row.get('Length', 100))
    if length <= 0:
        length = 100

    # Get a random starting position from the sequence
    start_pos = random.randint(0, max(0, len(seq) - 1))

    # Take 'length' number of positions from 'seq', wrapping around if necessary
    result = ''
    for offset in range(length):
        result += seq[(start_pos + offset) % len(seq)]

    return result

def orphan_insertions(row, reference_fasta):
    '''
    Function to generate orphan sequences
    '''
    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = transduction + polyA1 + TSD

    return seq

def Alu__FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR'
    for_value = safe_int_val(row['FOR'])
    # Get the Alu_Seq from the dictionary
    Alu_consensus = dict_consensus.get('Alu_Seq', '')

    # Slice the Alu_Seq from the end based on the 'FOR' value
    # Ensure that for_value is not greater than the length of Alu_Seq
    if isinstance(for_value, int) and for_value <= len(Alu_consensus):
        Alu_seq = Alu_consensus[-for_value:]
    else:
        # If for_value is larger than the length of the sequence, return the whole sequence
        Alu_seq = Alu_consensus

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = Alu_seq + polyA1 + TSD

    return seq

def L1__FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR'
    for_value = safe_int_val(row['FOR'])
    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    # Ensure that for_value is not greater than the length of L1_Seq
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        # If for_value is larger than the length of the sequence, return the whole sequence
        L1_seq = L1_consensus

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = L1_seq + polyA1 + TSD

    return seq

def L1__TD_FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR'
    for_value = safe_int_val(row['FOR'])
    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    # Ensure that for_value is not greater than the length of L1_Seq
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        # If for_value is larger than the length of the sequence, return the whole sequence
        L1_seq = L1_consensus

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = transduction + L1_seq + polyA1 + TSD

    return seq

def L1__FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR'
    for_value = safe_int_val(row['FOR'])
    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    # Ensure that for_value is not greater than the length of L1_Seq
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        # If for_value is larger than the length of the sequence, return the whole sequence
        L1_seq = L1_consensus

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = L1_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def L1__TRUN_REV_BLUNT_FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR' and 'REV' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + L1_seq + polyA1 + TSD

    return seq

def L1__TRUN_REV_BLUNT_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR' and 'REV' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + L1_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def L1__TRUN_REV_DUP_FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR', 'REV', and 'DUP' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])
    dup_value = safe_int_val(row['DUP'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Create DUP_seq by duplicating the 'DUP' bases of L1_seq
    if isinstance(dup_value, int) and dup_value > 0:
        dup_seq = L1_seq[:dup_value]
    else:
        dup_seq = ''  # If DUP value is invalid or 0

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + dup_seq + L1_seq + polyA1 + TSD

    return seq

def L1__TRUN_REV_DUP_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR', 'REV', and 'DUP' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])
    dup_value = safe_int_val(row['DUP'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on the 'FOR' value
    if isinstance(for_value, int) and for_value <= len(L1_consensus):
        L1_seq = L1_consensus[-for_value:]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Create DUP_seq by duplicating the 'DUP' bases of L1_seq
    if isinstance(dup_value, int) and dup_value > 0:
        dup_seq = L1_seq[:dup_value]
    else:
        dup_seq = ''  # If DUP value is invalid or 0

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + dup_seq + L1_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def L1__TRUN_REV_DEL_FOR_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR', 'REV', and 'DUP' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])
    del_value = safe_int_val(row['DEL'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on 'FOR' - 'DEL'
    if isinstance(for_value, int) and for_value > del_value and (for_value - del_value) <= len(L1_consensus):
        L1_seq = L1_consensus[-(for_value - del_value):]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + L1_seq + polyA1 + TSD

    return seq

def L1__TRUN_REV_DEL_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Extract the value from the 'FOR', 'REV', and 'DUP' columns
    for_value = safe_int_val(row['FOR'])
    rev_value = safe_int_val(row['REV'])
    del_value = safe_int_val(row['DEL'])

    # Get the L1_Seq from the dictionary
    L1_consensus = dict_consensus.get('L1_Seq', '')

    # Slice the L1_Seq from the end based on 'FOR' - 'DEL'
    if isinstance(for_value, int) and for_value > del_value and (for_value - del_value) <= len(L1_consensus):
        L1_seq = L1_consensus[-(for_value - del_value):]
    else:
        L1_seq = L1_consensus

    # Get the REV_seq based on the 'REV' value
    if isinstance(rev_value, int) and rev_value <= len(L1_consensus):
        # Slice the sequence from the end based on the 'REV' value starting after L1_seq
        rev_seq_start = len(L1_consensus) - for_value  # The point where L1_seq ends
        rev_seq = L1_consensus[rev_seq_start - rev_value: rev_seq_start]
        rev_seq = reverse_complementary(rev_seq)  # Apply reverse complementary to REV sequence
    else:
        rev_seq = ''  # In case REV value is invalid or too large

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = rev_seq + L1_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def SVA__SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # Final sequence
    seq = SINE_R_seq + polyA1 + TSD

    return seq

def SVA__VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Final sequence
    seq = vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    Alu_like_seq = dict_consensus.get('SVA_Alu-like_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Final sequence
    seq = Alu_like_seq + vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__MAST2_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    MAST2_seq = dict_consensus.get('SVA_MAST2_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Final sequence
    seq = MAST2_seq + vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__TD_MAST2_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    MAST2_seq = dict_consensus.get('SVA_MAST2_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = transduction + MAST2_seq + vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)


    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = SINE_R_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def SVA__VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = vntr_sequence + SINE_R_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def SVA__Alu_like_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    Alu_like_seq = dict_consensus.get('SVA_Alu-like_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = Alu_like_seq + vntr_sequence + SINE_R_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def SVA__MAST2_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    MAST2_seq = dict_consensus.get('SVA_MAST2_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = MAST2_seq + vntr_sequence + SINE_R_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def SVA__Hexamer_Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    Alu_like_seq = dict_consensus.get('SVA_Alu-like_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Hexamer
    hexamer = 'CCCTCT'
    sva_hexamer_length = safe_int_val(row['SVA_Hexamer'])  # Desired length of hexamer
    # Repeat the hexamer until we exceed or match the required length, then slice to get the exact length
    hexamer_seq = hexamer * (sva_hexamer_length // 6) + hexamer[:sva_hexamer_length % 6]

    # Final sequence
    seq = hexamer_seq + Alu_like_seq + vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__TD_Hexamer_Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    Alu_like_seq = dict_consensus.get('SVA_Alu-like_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Hexamer
    hexamer = 'CCCTCT'
    sva_hexamer_length = safe_int_val(row['SVA_Hexamer'])  # Desired length of hexamer
    # Repeat the hexamer until we exceed or match the required length, then slice to get the exact length
    hexamer_seq = hexamer * (sva_hexamer_length // 6) + hexamer[:sva_hexamer_length % 6]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = transduction + hexamer_seq + Alu_like_seq + vntr_sequence + SINE_R_seq + polyA1 + TSD

    return seq

def SVA__Hexamer_Alu_like_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list):
    # Get the L1_Seq from the dictionary
    SINE_R_seq = dict_consensus.get('SVA_SINE-R_Seq', '')
    Alu_like_seq = dict_consensus.get('SVA_Alu-like_Seq', '')

    # Poly A 1
    if pd.isna(row['PolyA_Length_1']) or row['PolyA_Length_1'] == 'NA':
        polyA_length1 = 0
    else:
        polyA_length1 = safe_int_val(row['PolyA_Length_1'])
    polyA1 = 'A' * polyA_length1

    # Poly A 2
    if pd.isna(row['PolyA_Length_2']) or row['PolyA_Length_2'] == 'NA':
        polyA_length2 = 0
    else:
        polyA_length2 = safe_int_val(row['PolyA_Length_2'])
    polyA2 = 'A' * polyA_length2

    # TSD
    beg = safe_int_val(row['beg'], 1000)
    tsd_len = safe_int_val(row['TSD_Length'])
    start_tsd = beg - tsd_len
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)

    # VNTR
    random_row = random.choice(SVA_VNTR_list).strip()  # Select random VNTR motif
    sva_vntr_length = safe_int_val(row['SVA_VNTR_Length'])  # Desired length of VNTR

    # Generate VNTR sequence taking desired bases
    # In case VNTR is smaller than desired bases, repeat and wrap around the selected sequence
    vntr_sequence = ''
    while len(vntr_sequence) < sva_vntr_length:
        vntr_sequence += random_row
    vntr_sequence = vntr_sequence[:sva_vntr_length]

    # Hexamer
    hexamer = 'CCCTCT'
    sva_hexamer_length = safe_int_val(row['SVA_Hexamer'])  # Desired length of hexamer
    # Repeat the hexamer until we exceed or match the required length, then slice to get the exact length
    hexamer_seq = hexamer * (sva_hexamer_length // 6) + hexamer[:sva_hexamer_length % 6]

    # Transduction
    # Fetch a sequence using pysam
    with pysam.FastaFile(reference_fasta) as fasta_file:
        transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])

    # Final sequence
    seq = hexamer_seq + Alu_like_seq + vntr_sequence + SINE_R_seq + polyA1 + transduction + polyA2 + TSD

    return seq

def generate_insertion_seq(row, motifs_file, reference_fasta, dict_consensus, SVA_VNTR_list):
    '''
    Global function that for each row of the df generates the determied sequence
    '''

    # VNTR
    if row['name'] == 'VNTR':
        sequence, selected_motifs, vntr_num_motifs = VNTR_insertions(row, motifs_file)
        return sequence, selected_motifs, vntr_num_motifs

    # DUPLICATIONS
    elif row['name'] == 'DUP':
        return DUP_insertions(row, reference_fasta), 0, 0

    # NUMT
    elif row['name'] == 'NUMT':
        return NUMT_insertions(row, dict_consensus), 0, 0

    # INVERSE DUPLICATIONS
    elif row['name'] == 'INV_DUP':
        seq = DUP_insertions(row, reference_fasta)
        return reverse_complementary(seq), 0, 0

    # ORPHAN
    elif row['name'] == 'orphan':
        return orphan_insertions(row, reference_fasta)   , 0, 0

    # Alu__FOR+POLYA
    elif row['name'] == 'Alu__FOR+POLYA':
        return Alu__FOR_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # Alu__TRUN+FOR+POLYA
    elif row['name'] == 'Alu__TRUN+FOR+POLYA':
        return Alu__FOR_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # L1__FOR+POLYA
    elif row['name'] == 'L1__FOR+POLYA':
        return L1__FOR_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # L1__TRUN+FOR+POLYA
    elif row['name'] == 'L1__TRUN+FOR+POLYA':
        return L1__FOR_POLYA_insertions(row, dict_consensus, reference_fasta) , 0  , 0

    # L1__TD+FOR+POLYA
    elif row['name'] == 'L1__TD+FOR+POLYA':
        return L1__TD_FOR_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # L1__FOR+POLYA+TD+POLYA
    elif row['name'] == 'L1__FOR+POLYA+TD+POLYA':
        return L1__FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # L1__TRUN+FOR+POLYA+TD+POLYA
    elif row['name'] == 'L1__TRUN+FOR+POLYA+TD+POLYA':
        return L1__FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # L1__TRUN+REV+BLUNT+FOR+POLYA
    elif row['name'] == 'L1__TRUN+REV+BLUNT+FOR+POLYA':
        return L1__TRUN_REV_BLUNT_FOR_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA
    elif row['name'] == 'L1__TRUN+REV+BLUNT+FOR+POLYA+TD+POLYA':
        return L1__TRUN_REV_BLUNT_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__TRUN+REV+DUP+FOR+POLYA
    elif row['name'] == 'L1__TRUN+REV+DUP+FOR+POLYA':
        return L1__TRUN_REV_DUP_FOR_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__TRUN+REV+DUP+FOR+POLYA+TD+POLYA
    elif row['name'] == 'L1__TRUN+REV+DUP+FOR+POLYA+TD+POLYA':
        return L1__TRUN_REV_DUP_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__TRUN+REV+DEL+FOR+POLYA
    elif row['name'] == 'L1__TRUN+REV+DEL+FOR+POLYA':
        return L1__TRUN_REV_DEL_FOR_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__REV+DEL+FOR+POLYA
    elif row['name'] == 'L1__REV+DEL+FOR+POLYA':
        return L1__TRUN_REV_DEL_FOR_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # L1__TRUN+REV+DEL+FOR+POLYA+TD+POLYA
    elif row['name'] == 'L1__TRUN+REV+DEL+FOR+POLYA+TD+POLYA':
        return L1__TRUN_REV_DEL_FOR_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # SVA__SINE-R+POLYA
    elif row['name'] == 'SVA__SINE-R+POLYA':
        return SVA__SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta) , 0, 0

    # SVA__VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__VNTR+SINE-R+POLYA':
        return SVA__VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)   , 0 , 0

    # SVA__Alu-like+VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__Alu-like+VNTR+SINE-R+POLYA':
        return SVA__Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)    , 0, 0

    # SVA__MAST2+VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__MAST2+VNTR+SINE-R+POLYA':
        return SVA__MAST2_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)  , 0, 0

    # SVA__TD+MAST2+VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__TD+MAST2+VNTR+SINE-R+POLYA':
        return SVA__TD_MAST2_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)  , 0  , 0

    # SVA__SINE-R+POLYA+TD+POLYA
    elif row['name'] == 'SVA__SINE-R+POLYA+TD+POLYA':
        return SVA__SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta)   , 0, 0

    # SVA__VNTR+SINE-R+POLYA+TD+POLYA
    elif row['name'] == 'SVA__VNTR+SINE-R+POLYA+TD+POLYA':
        return SVA__VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list), 0, 0

    # SVA__Alu-like+VNTR+SINE-R+POLYA+TD+POLYA
    elif row['name'] == 'SVA__Alu-like+VNTR+SINE-R+POLYA+TD+POLYA':
        return SVA__Alu_like_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list), 0, 0

    # SVA__MAST2+VNTR+SINE-R+POLYA+TD+POLYA
    elif row['name'] == 'SVA__MAST2+VNTR+SINE-R+POLYA+TD+POLYA':
        return SVA__MAST2_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)  , 0, 0

    # SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA':
        return SVA__Hexamer_Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)    , 0, 0

    # SVA__TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA
    elif row['name'] == 'SVA__TD+Hexamer+Alu-like+VNTR+SINE-R+POLYA':
        return SVA__TD_Hexamer_Alu_like_VNTR_SINE_R_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)    , 0, 0

    # SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA
    elif row['name'] == 'SVA__Hexamer+Alu-like+VNTR+SINE-R+POLYA+TD+POLYA':
        return SVA__Hexamer_Alu_like_VNTR_SINE_R_POLYA_TD_POLYA_insertions(row, dict_consensus, reference_fasta, SVA_VNTR_list)   , 0 , 0

    else:
        return '', 0, 0

def reverse_complementary(seq):
    '''
    Function to create reverse complementary of a given sequence
    '''
    # initialize empty comp list to store complementary sequence
    comp = []
    seq = seq.upper()
    for base in seq:
        if base == 'A':
            comp.append('T')
        elif base == 'T':
            comp.append('A')
        elif base == 'G':
            comp.append('C')
        elif base == 'C':
            comp.append('G')
    # start counter with the length of the complementary sequence
    counter = len(comp) - 1
    # empty list to start the reversee
    reverse = []
    # while the counter is not 0
    while counter >= 0:
        # take the last position of the complementary sequence
        base = comp[counter]
        # add it to the rerverse list
        reverse.append(base)
        # jump to the previous position
        counter -= 1
    # join the list
    reverse_sequence = ''.join(reverse)
    return reverse_sequence

def RC_insertion(df_insertions):
    '''
    Function to create the reverse complementary of those new generated sequences in the negative strand
    '''
    df_insertions['Sequence_Insertion'] = df_insertions.apply(
        lambda row: reverse_complementary(row['Sequence_Insertion']) if row['Strand'] == '-' else row['Sequence_Insertion'],
        axis=1
    )
    return df_insertions

def update_sequences(df_insertions):
    # Add a new column with 'Insertion' in each row
    df_insertions['Event_Type'] = 'Insertion'
    # Write all sequences in upper case
    df_insertions['Sequence_Insertion'] = df_insertions['Sequence_Insertion'].str.upper()

    # For those in - strand apply reverse complementary
    df_insertions = RC_insertion(df_insertions)

    return df_insertions

# Function to process each row's 'name' value
def process_name(name):
    if '__' in name:  # Case when there is '__' in the name
        fam_n, conformation = name.split('__', 1)
        itype_n = 'partnered' if 'TD' in conformation else 'solo'
    else:  # Case when there is no '__' in the name
        fam_n, conformation = np.nan, np.nan
        itype_n = name

    return pd.Series([itype_n, conformation, fam_n])

# Function to generate the HEXAMER_SEQ column
def generate_hexamer_seq(row):
    # Check if 'FAM_N' contains 'SVA' and 'CONFORMATION' contains 'Hexamer'
    if pd.notna(row['FAM_N']) and 'SVA' in row['FAM_N'] and pd.notna(row['CONFORMATION']) and 'Hexamer' in row['CONFORMATION']:
        hexamer = 'CCCTCT'
        try:
            # Assuming HEXAMER_LEN is a column with the desired hexamer length
            sva_hexamer_length = safe_int_val(row['HEXAMER_LEN'])  # Desired length of hexamer
            hexamer_seq = hexamer * (sva_hexamer_length // 6) + hexamer[:sva_hexamer_length % 6]
            return hexamer_seq
        except (ValueError, TypeError):
            return np.nan  # Return NaN if there's an issue with the length
    return np.nan  # Return NaN if conditions are not met

# Function to generate TSD sequence from the reference fasta file
def generate_tsd_seq(row, reference_fasta):
    # Check if 'beg' and 'TSD_LEN' are valid
    if pd.notna(row['beg']) and pd.notna(row['TSD_LEN']):
        try:
            beg = safe_int_val(row['beg'], 1000)
            tsd_len = safe_int_val(row['TSD_LEN'])
            start_tsd = beg - tsd_len

            # Fetch the sequence from the reference FASTA file using pysam
            with pysam.FastaFile(reference_fasta) as fasta_file:
                TSD = fasta_file.fetch(row['#ref'], start_tsd, beg)  # Extract the sequence
                return TSD.upper()
        except (ValueError, TypeError, KeyError):
            return np.nan  # Return NaN if there's an issue with the length or values
    return np.nan  # Return NaN if either 'beg' or 'TSD_LEN' is missing or invalid

# Function to generate sequence for 3PRIME_TD
def generate_3prime_td_seq(row, reference_fasta):
    # Fetch the sequence for 3PRIME_TD_SEQ if there's a value in 3PRIME_NB_TD
    if pd.notna(row['3PRIME_NB_TD']):
        try:
            with pysam.FastaFile(reference_fasta) as fasta_file:
                transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])
                return transduction
        except (KeyError, ValueError):
            return np.nan
    return np.nan

# Function to generate sequence for 5PRIME_TD
def generate_5prime_td_seq(row, reference_fasta):
    # Fetch the sequence for 5PRIME_TD_SEQ if there's a value in 5PRIME_NB_TD
    if pd.notna(row['5PRIME_NB_TD']):
        try:
            with pysam.FastaFile(reference_fasta) as fasta_file:
                transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])
                return transduction
        except (KeyError, ValueError):
            return np.nan
    return np.nan

# Function to generate sequence for orphan TD
def generate_orphan_seq(row, reference_fasta):
    # Fetch the sequence for orphan TD if there's a value in ORPHAN_TD_COORD
    if pd.notna(row['ORPHAN_TD_COORD']):
        try:
            with pysam.FastaFile(reference_fasta) as fasta_file:
                transduction = fasta_file.fetch(row['SRC_ref'], row['TD_beg'], row['TD_end'])
                return transduction
        except (KeyError, ValueError):
            return np.nan
    return np.nan

# Function to generate POLYA_LEN column
def generate_polya_len(row):
    # Remove the decimal by converting the value to an integer before adding to the string
    poly_a_len_1 = str(safe_int_val(row['PolyA_Length_1'])) if pd.notna(row['PolyA_Length_1']) else ''
    poly_a_len_2 = str(safe_int_val(row['PolyA_Length_2'])) if pd.notna(row['PolyA_Length_2']) else ''

    if poly_a_len_1 and poly_a_len_2:
        return poly_a_len_1 + ',' + poly_a_len_2
    elif poly_a_len_1:
        return poly_a_len_1
    else:
        return np.nan  # Return NaN if both are missing

# Function to generate POLYA_SEQ column
def generate_polya_seq(row):
    strand = row['STRAND'] if pd.notna(row['STRAND']) else '+'

    # Determine the PolyA sequences
    seq_1 = 'A' * safe_int_val(row['PolyA_Length_1']) if pd.notna(row['PolyA_Length_1']) else ''
    seq_2 = 'A' * safe_int_val(row['PolyA_Length_2']) if pd.notna(row['PolyA_Length_2']) else ''

    if strand == '-':
        seq_1 = seq_1.replace('A', 'T') if seq_1 else ''
        seq_2 = seq_2.replace('A', 'T') if seq_2 else ''

    # Join the sequences with a comma if both exist
    if seq_1 and seq_2:
        return seq_1 + ',' + seq_2
    elif seq_1:
        return seq_1
    elif seq_2:
        return seq_2
    return np.nan  # Return NaN if no sequences are available

def process_vntr_motifs(df):
    # Iterate over each element in the 'Selected_VNTR_Motifs' column
    for i in range(len(df)):
        #motif = df.at[i, 'Selected_VNTR_Motifs']
        motif = df.iloc[i]['Selected_VNTR_Motifs']
        if motif == '0':
            df.at[i, 'Selected_VNTR_Motifs'] = np.nan  # Replace '0' with NaN
        elif isinstance(motif, str):
            # Remove '[' and ']' from the string
            df.at[i, 'Selected_VNTR_Motifs'] = motif.replace("[", "").replace("'[", "").replace("]", "").replace("]'", "").strip()

    return df

# Main function
def df_VCF(df, reference_fasta):
    # Create a copy of the DataFrame
    df_copy = df.copy()
    df_copy = process_vntr_motifs(df_copy)
    df_copy['Selected_VNTR_Motifs'] = df_copy['Selected_VNTR_Motifs'].str.replace("'", "")
    # Rename columns based on your requirements
    df_copy = df_copy.rename(columns={
        'Length': 'INS_LEN',
        'VNTR_Num_Motifs': 'NB_MOTIFS',
        'SVA_Hexamer': 'HEXAMER_LEN',
        'TSD_Length': 'TSD_LEN',
        'TD_5': '5PRIME_TD_LEN',
        'TD_3': '3PRIME_TD_LEN',
        'Strand': 'STRAND',
        'TD_orphan_Length': 'ORPHAN_TD_LEN',
        'Selected_VNTR_Motifs': 'MOTIFS'
    })

    # Convert columns to integers to remove decimals (i.e., .0)
    df_copy['NB_MOTIFS'] = df_copy['NB_MOTIFS'].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('Int64')  # Use 'Int64' to keep NaNs
    df_copy['HEXAMER_LEN'] = df_copy['HEXAMER_LEN'].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('Int64')  # Use 'Int64' to keep NaNs
    df_copy['TSD_LEN'] = df_copy['TSD_LEN'].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('Int64')  # Use 'Int64' to keep NaNs
    df_copy['5PRIME_TD_LEN'] = df_copy['5PRIME_TD_LEN'].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('Int64')  # Use 'Int64' to keep NaNs
    df_copy['3PRIME_TD_LEN'] = df_copy['3PRIME_TD_LEN'].apply(lambda x: int(x) if pd.notna(x) else np.nan).astype('Int64')  # Use 'Int64' to keep NaNs

    # Add the 'ID' column with the format SV1, SV2, SV3...
    df_copy['ID'] = ['INS_' + str(i + 1) for i in range(len(df_copy))]

    # Apply the processing function to the 'name' column and split it into three new columns
    df_copy[['ITYPE_N', 'CONFORMATION', 'FAM_N']] = df_copy['name'].apply(process_name)

    # For rows where there is no 'CONFORMATION' (e.g., VNTR), set 'CONFORMATION' to NaN
    df_copy['CONFORMATION'] = df_copy['CONFORMATION'].replace('', np.nan)

    # Remove the 'name' column at the end
    df_copy = df_copy.drop(columns=['name'])

    # Apply the hexamer sequence generation logic
    df_copy['HEXAMER_SEQ'] = df_copy.apply(generate_hexamer_seq, axis=1)

    # Apply the TSD sequence generation logic
    df_copy['TSD_SEQ'] = df_copy.apply(lambda row: generate_tsd_seq(row, reference_fasta), axis=1)

    # Create the 3PRIME_NB_TD and 5PRIME_NB_TD columns based on the conditions
    df_copy['3PRIME_NB_TD'] = df_copy['3PRIME_TD_LEN'].apply(lambda x: 1 if pd.notna(x) else np.nan)
    df_copy['5PRIME_NB_TD'] = df_copy['5PRIME_TD_LEN'].apply(lambda x: 1 if pd.notna(x) else np.nan)

    # Create the 3PRIME_TD_COORD, 5PRIME_TD_COORD and ORPHAN_TD_COORD columns based on the conditions
    df_copy['3PRIME_TD_COORD'] = df_copy['SRC_identifier'].where(df_copy['3PRIME_NB_TD'].notna(), np.nan)
    df_copy['5PRIME_TD_COORD'] = df_copy['SRC_identifier'].where(df_copy['5PRIME_NB_TD'].notna(), np.nan)
    df_copy['ORPHAN_TD_COORD'] = df_copy['SRC_identifier'].where(df_copy['ITYPE_N'] == 'orphan')

    # Remove the SRC_identifier column
    df_copy = df_copy.drop(columns=['SRC_identifier'])

    # Apply the transduction sequence generation logic for 3PRIME_TD_SEQ and 5PRIME_TD_SEQ
    df_copy['3PRIME_TD_SEQ'] = df_copy.apply(lambda row: generate_3prime_td_seq(row, reference_fasta), axis=1)
    df_copy['5PRIME_TD_SEQ'] = df_copy.apply(lambda row: generate_5prime_td_seq(row, reference_fasta), axis=1)
    df_copy['ORPHAN_TD_SEQ'] = df_copy.apply(lambda row: generate_orphan_seq(row, reference_fasta), axis=1)

    # Convert the 3PRIME_TD_SEQ and 5PRIME_TD_SEQ columns to uppercase
    df_copy['3PRIME_TD_SEQ'] = df_copy['3PRIME_TD_SEQ'].apply(lambda x: x.upper() if pd.notna(x) else np.nan)
    df_copy['5PRIME_TD_SEQ'] = df_copy['5PRIME_TD_SEQ'].apply(lambda x: x.upper() if pd.notna(x) else np.nan)
    df_copy['ORPHAN_TD_SEQ'] = df_copy['ORPHAN_TD_SEQ'].apply(lambda x: x.upper() if pd.notna(x) else np.nan)

    # Apply the POLYA_LEN generation logic
    df_copy['POLYA_LEN'] = df_copy.apply(generate_polya_len, axis=1)

    # Apply the POLYA_SEQ generation logic
    df_copy['POLYA_SEQ'] = df_copy.apply(generate_polya_seq, axis=1)

    return df_copy

def create_vcf_file(df, reference_fasta, chromosome_length):
    # Create the chr_length dictionary using formats
    chr_length = formats.chrom_lengths_index(chromosome_length)

    # Get date of creation
    current_date = datetime.datetime.now().strftime("%Y%m%d")
    reference_name = os.path.basename(reference_fasta)

    # Order chromosomes
    def sort_chromosomes(contig):
        contig_clean = contig.lower().replace("chr", "")
        special_order = {"x": 23, "y": 24, "m": 25, "mt": 25}

        if contig_clean.isdigit():
            return (0, int(contig_clean))

        if contig_clean in special_order:
            return (0, special_order[contig_clean])

        return (1, contig_clean)

    contigs = sorted(df['#ref'].unique(), key=sort_chromosomes)

    # Open a VCF file to write to
    with open('VCF_Insertions_SVModeller.vcf', 'w') as vcf_file:
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write(f"##fileDate={current_date}\n")
        vcf_file.write("##source=SVModeller\n")
        vcf_file.write(f"##reference={reference_name}\n")

        # Adding contig length
        for contig in contigs:
            if contig in chr_length:
                contig_length = chr_length[contig]
                vcf_file.write(f"##contig=<ID={contig},assembly=None,length={contig_length},species=None>\n")
            else:
                vcf_file.write(f"##contig=<ID={contig},assembly=None,length=None,species=None>\n")

        vcf_file.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        vcf_file.write("##INFO=<ID=INS_LEN,Type=float,Description=Total length of the insertion>\n")
        vcf_file.write("##INFO=<ID=TSD_LEN,Type=float,Description=Length of the TSD>\n")
        vcf_file.write("##INFO=<ID=5PRIME_TD_LEN,Type=float,Description=Length of the 5' TD>\n")
        vcf_file.write("##INFO=<ID=3PRIME_TD_LEN,Type=float,Description=Length of the 3' TD>\n")
        vcf_file.write("##INFO=<ID=NB_MOTIFS,Type=float,Description=Number of VNTR motifs>\n")
        vcf_file.write("##INFO=<ID=MOTIFS,Type=str,Description=VNTR selected motifs>\n")
        vcf_file.write("##INFO=<ID=HEXAMER_LEN,Type=float,Description=Length of the SVA hexamer>\n")
        vcf_file.write("##INFO=<ID=SVA_VNTR_Length,Type=float,Description=Length of the SVA VNTR>\n")
        vcf_file.write("##INFO=<ID=ITYPE_N,Type=float,Description=Type of insertion>\n")
        vcf_file.write("##INFO=<ID=CONFORMATION,Type=float,Description=Conformation of the insertion>\n")
        vcf_file.write("##INFO=<ID=FAM_N,Type=float,Description=Family of the insertion>\n")
        vcf_file.write("##INFO=<ID=HEXAMER_SEQ,Type=float,Description=Sequence of SVA hexamer>\n")
        vcf_file.write("##INFO=<ID=TSD_SEQ,Type=float,Description=Length of the poly A tail (or first poly A in case of more than 1)>\n")
        vcf_file.write("##INFO=<ID=3PRIME_NB_TD,Type=float,Description=Number of 3' TD>\n")
        vcf_file.write("##INFO=<ID=5PRIME_NB_TD,Type=float,Description=Number of 5' TD>\n")
        vcf_file.write("##INFO=<ID=3PRIME_TD_COORD,Type=float,Description=Coordinates of 3' TD>\n")
        vcf_file.write("##INFO=<ID=5PRIME_TD_COORD,Type=float,Description=Coordinates of 5' TD>\n")
        vcf_file.write("##INFO=<ID=3PRIME_TD_SEQ,Type=float,Description=Sequence of 3' TD>\n")
        vcf_file.write("##INFO=<ID=5PRIME_TD_SEQ,Type=float,Description=Sequence of 5' TD>\n")
        vcf_file.write("##INFO=<ID=POLYA_LEN,Type=float,Description=Length of the poly A tail>\n")
        vcf_file.write("##INFO=<ID=POLYA_SEQ,Type=float,Description=Sequence of the poly A tail>\n")
        vcf_file.write("##INFO=<ID=FOR,Type=float,Description=Length of conformation forward part of the event>\n")
        vcf_file.write("##INFO=<ID=TRUN,Type=float,Description=Length of the conformation truncated part of the event>\n")
        vcf_file.write("##INFO=<ID=REV,Type=float,Description=Length of the conformation reverse part of the event>\n")
        vcf_file.write("##INFO=<ID=DEL,Type=float,Description=Length of the conformation deleted part of the event>\n")
        vcf_file.write("##INFO=<ID=DUP,Type=float,Description=Length of the conformation duplicated part of the event>\n")
        vcf_file.write("##INFO=<ID=STRAND,Type=str,Description=Strand (+ or -) of the event>\n")
        vcf_file.write("##INFO=<ID=ORPHAN_TD_LEN,Type=float,Description=Length of the orphan transduction>\n")
        vcf_file.write("##INFO=<ID=ORPHAN_TD_COORD,Type=str,Description=Coordinates of orphan transduction>\n")
        vcf_file.write("##INFO=<ID=ORPHAN_TD_SEQ,Type=str,Description=Sequence of orphan transduction>\n")

        vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        with pysam.FastaFile(reference_fasta) as fasta_file:

            for _, row in df.iterrows():
                chrom = row['#ref']
                pos = row['beg']
                event_id = row['ID']
                alt = row['Sequence_Insertion']
                qual = '.'
                filter_val = '.'

                beg = int(pos)

                # Fetch reference base
                ref = fasta_file.fetch(chrom, beg - 1, beg).upper()

                info_fields = []
                for col in ['ITYPE_N', 'INS_LEN', 'STRAND', 'FAM_N', 'CONFORMATION',
                            'FOR', 'TRUN', 'REV', 'DEL', 'DUP', 'TSD_LEN', 'TSD_SEQ',
                            '3PRIME_NB_TD', '5PRIME_NB_TD', '5PRIME_TD_LEN',
                            '3PRIME_TD_LEN', '5PRIME_TD_COORD', '3PRIME_TD_COORD',
                            '3PRIME_TD_SEQ', '5PRIME_TD_SEQ', 'NB_MOTIFS',
                            'MOTIFS', 'HEXAMER_LEN', 'SVA_VNTR_Length',
                            'HEXAMER_SEQ', 'ORPHAN_TD_LEN', 'ORPHAN_TD_COORD',
                            'ORPHAN_TD_SEQ', 'POLYA_LEN', 'POLYA_SEQ']:

                    value = row[col]
                    if pd.notna(value):
                        info_fields.append(f"{col}={value}")

                info_fields.append("SVTYPE=INS")

                info = ";".join(info_fields)

                vcf_file.write(f"{chrom}\t{pos}\t{event_id}\t{ref}\t{alt}\t{qual}\t{filter_val}\t{info}\n")

    with open('VCF_Insertions_SVModeller.vcf', 'r') as file:
        content = file.read()

    content = content.replace("5PRIME_TD_LEN", "TD_LEN_5PRIME")
    content = content.replace("3PRIME_TD_LEN", "TD_LEN_3PRIME")
    content = content.replace("3PRIME_NB_TD", "NB_TD_3PRIME")
    content = content.replace("5PRIME_NB_TD", "NB_TD_5PRIME")
    content = content.replace("3PRIME_TD_COORD", "TD_COORD_3PRIME")
    content = content.replace("5PRIME_TD_COORD", "TD_COORD_5PRIME")
    content = content.replace("3PRIME_TD_SEQ", "TD_SEQ_3PRIME")
    content = content.replace("5PRIME_TD_SEQ", "TD_SEQ_5PRIME")
    content = content.replace("INS_LEN", "SVLEN")

    with open('VCF_Insertions_SVModeller.vcf', 'w') as file:
        file.write(content)

    print("VCF file created successfully.")

def classify_mutations_in_bins(chromosome_length, bin_size, merged_df):
    """
    Classify mutations into genomic bins based on chromosome length and bin size.

    Parameters:
    chromosome_length (dict): Dictionary with chromosome lengths.
    bin_size (int): The size of each genomic bin.
    merged_df: DataFrame containing mutation data.

    Returns:
    pandas.DataFrame: Classified mutations for each genomic bin.
    """
    # Get chromosome lengths
    chr_length = formats.chrom_lengths_index(chromosome_length)

    # Generate bins for the genome dynamically based on chromosome_length file
    chromosomes = list(chr_length.keys())
    bins = gRanges.makeGenomicBins(chr_length, bin_size, chromosomes)[::-1]

    # Classify mutations in each window of the genomic bins
    res_table = mut_bins(bins, merged_df)

    return res_table

def add_columns(df1, df2, df3):
    '''
    Function to add start and end columns to the df based on probabilities
    '''
    new_df = pd.DataFrame(columns=['#ref', 'beg', 'end', 'Length'])
    for _, row in df3.iterrows():
        name = str(row['Event'])
        base_name = name.split('__')[0]

        if name in df2.columns:
            prob_df = df2[df2[name] > 0]
        elif base_name in df2.columns:
            prob_df = df2[df2[base_name] > 0]
        else:
            prob_df = pd.DataFrame()

        if prob_df.empty:
            name_matches = df1[df1['Event'] == name]
            if name_matches.empty:
                name_matches = df1[df1['Event'] == base_name]
            if name_matches.empty:
                name_matches = df1
            random_row = name_matches.sample(n=1)
        else:
            name_df = df1[df1['Event'] == name]
            if name_df.empty:
                name_df = df1[df1['Event'] == base_name]
            if name_df.empty:
                name_df = df1

            target_col = name if name in df2.columns else base_name
            weights = prob_df[target_col].values.astype(float)
            if weights.sum() == 0:
                weights = np.ones(len(weights)) / len(weights)
            else:
                weights = weights / weights.sum()
            weights = np.repeat(weights, len(name_df) // len(weights) + 1)[:len(name_df)]
            if weights.sum() == 0:
                weights = np.ones(len(name_df)) / len(name_df)
            else:
                weights = weights / weights.sum()
            random_row = name_df.sample(n=1, weights=weights)

        new_df = pd.concat([new_df, random_row[['#ref', 'beg', 'end', 'Length']]], ignore_index=True)

    df3[['#ref', 'beg', 'end', 'Length']] = new_df[['#ref', 'beg', 'end', 'Length']]
    return df3

def generate_deletion_events(probabilities_table, num_events, deletions_table, genome_wide_distribution):
    # Generate name of the events based on their proportions
    probs = pd.to_numeric(probabilities_table['Probability'], errors='coerce').fillna(0).values.astype(float)
    if probs.sum() == 0:
        p_norm = np.ones(len(probs)) / len(probs)
    else:
        p_norm = probs / probs.sum()
    sampled_names = np.random.choice(probabilities_table['Event'], size=num_events, p=p_norm)
    table_events = pd.DataFrame({'Event': sampled_names})

    # Add the info for the deletions
    df3 = add_columns(deletions_table, genome_wide_distribution, table_events)

    # Add a new column 'Event_Type' with the value 'insertion'
    df3['Event_Type'] = 'Deletion'

    # Define the new order of columns first
    new_order = ['#ref', 'beg', 'end', 'Event_Type', 'Event', 'Length']

    # Update order and name of columns
    df3 = df3[new_order]

    return df3

# Function to process the df of deletion events and create the VCF
def create_VCF(df, reference_fasta, chromosome_length):
    # Rename the column 'Length' to 'DEL_LEN'
    df = df.rename(columns={'Length': 'DEL_LEN', 'name': 'DTYPE_N'})

    # Create a new empty column 'Sequence'
    df['Sequence'] = None

    # Create a new column 'Seq_end' which is the sum of 'beg' + 'DEL_LEN'
    df['beg'] = pd.to_numeric(df['beg'], errors='coerce')
    df['DEL_LEN'] = pd.to_numeric(df['DEL_LEN'], errors='coerce')

    df['Seq_end'] = df['beg'] + df['DEL_LEN']

    # Now, for each row, fetch the sequence using pysam and write to 'Sequence' column
    for index, row in df.iterrows():
        # Ensure 'beg' and 'Seq_end' are valid integers
        beg = safe_int_val(row['beg'], 1000)
        end = safe_int_val(row['Seq_end'], beg + 100)

        # Fetch the sequence using pysam
        with pysam.FastaFile(reference_fasta) as fasta_file:
            TSD = fasta_file.fetch(row['#ref'], beg, end)

        # Store the result in the 'Sequence' column
        df.at[index, 'Sequence'] = TSD

    # Convert the 'Sequence' column to uppercase
    df['Sequence'] = df['Sequence'].str.upper()

    df['ID'] = ['DEL_' + str(i + 1) for i in range(len(df))]

    # Create the chr_length dictionary using formats
    chr_length = formats.chrom_lengths_index(chromosome_length)

    # Get date of creation
    current_date = datetime.datetime.now().strftime("%Y%m%d")

    reference_name = os.path.basename(reference_fasta)

    # Order chromosomes
    def sort_chromosomes(contig):
        contig_clean = contig.lower().replace("chr", "")
        special_order = {"x": 23, "y": 24, "m": 25, "mt": 25}
        if contig_clean.isdigit():
            return (0, int(contig_clean))
        if contig_clean in special_order:
            return (0, special_order[contig_clean])
        return (1, contig_clean)

    contigs = sorted(df['#ref'].unique(), key=sort_chromosomes)

    # Open a VCF file to write to
    with open('VCF_Deletions_SVModeller.vcf', 'w') as vcf_file:
        # Write VCF header
        vcf_file.write("##fileformat=VCFv4.2\n")
        vcf_file.write(f"##fileDate={current_date}\n")
        vcf_file.write("##source=SVModeller\n")
        vcf_file.write(f"##reference={reference_name}\n")

        # Adding contig length
        for contig in contigs:
            if contig in chr_length:
                contig_length = chr_length[contig]
                vcf_file.write(f"##contig=<ID={contig},assembly=None,length={contig_length},species=None>\n")
            else:
                vcf_file.write(f"##contig=<ID={contig},assembly=None,length=None,species=None>\n")

        vcf_file.write('##INFO=<ID=SVTYPE,Number=1,Type=String,Description="Type of structural variant">\n')
        vcf_file.write("##INFO=<ID=DEL_LEN,Type=float,Description=Total length of the deletion>\n")
        vcf_file.write("##INFO=<ID=DEL_N,Type=float,Description=Type of deletion>\n")
        vcf_file.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\n")

        # Loop through each row in the DataFrame
        for _, row in df.iterrows():
            chrom = row['#ref']
            pos = row['beg']
            event_id = row['ID']
            ref = row['Sequence']
            alt = '.'
            qual = '.'
            filter = '.'

            # Convert 'beg' to an integer (position at which to fetch the reference sequence)
            beg = safe_int_val(row['beg'], 1000)

            # Fetch the sequence using pysam from the reference genome at the position 'beg'
            with pysam.FastaFile(reference_fasta) as fasta_file:
                alt = fasta_file.fetch(chrom, beg - 1, beg)  # pysam is 0-based, so we subtract 1 for 0-based indexing
            alt = alt.upper()

            # Create the INFO field dynamically, excluding NaN values
            info_fields = []
            for col in ['DTYPE_N', 'DEL_LEN']:
                value = row[col]
                if pd.notna(value):  # Check if the value is not NaN
                    info_fields.append(f"{col}={value}")

            info_fields.append("SVTYPE=DEL")

            # Join the info fields into a single string separated by semicolons
            info = ";".join(info_fields)

            # Write the VCF entry for each row
            vcf_file.write(f"{chrom}\t{pos}\t{event_id}\t{ref}\t{alt}\t{qual}\t{filter}\t{info}\n")

    with open('VCF_Deletions_SVModeller.vcf', 'r') as file:
        content = file.read()

    content = content.replace("DEL_LEN", "SVLEN")

    with open('VCF_Deletions_SVModeller.vcf', 'w') as file:
        file.write(content)

    print("VCF file created successfully.")

    return df

def write_fasta(file_path, seq_dict):
    '''Create a FASTA file from a dictionary of sequences'''
    with open(file_path, 'w') as fasta_file:
        for header, sequence in seq_dict.items():
            fasta_file.write(f">{header}\n")
            fasta_file.write(f"{sequence}\n")

# Function to run PBSIM to generate synthetic reads for reference and modified genomes
def run_pbsim(genome, method_file, method, depth, output_dir, output_reference):
    if depth == 0:
        print("Depth is 0. Skipping PBSIM execution.")
        return

    method_file = os.path.abspath(method_file)
    output_prefix = os.path.join(output_dir, output_reference)

    if method == 'quality_score':
        command = f"pbsim --strategy wgs --method qshmm --qshmm {method_file} --depth {depth} --genome {genome} --prefix {output_prefix}"
    elif method == 'error_model':
        command = f"pbsim --strategy wgs --method errhmm --errhmm {method_file} --depth {depth} --genome {genome} --prefix {output_prefix}"
    elif method == 'training':
        command = f"pbsim --strategy wgs --method sample --sample {method_file} --depth {depth} --genome {genome} --prefix {output_prefix}"
    else:
        raise ValueError(f"Unknown method: {method}")

    print(f"Running PBSIM with command: {command}")
    subprocess.run(command, shell=True, check=True)

    return output_prefix

# Function to align reads using Minimap2
def run_minimap2(reference_file, fastq_file_1, fastq_file_2, output_bam, technology, threads):
    fastqs = fastq_file_1
    if fastq_file_2:
        fastqs += f" {fastq_file_2}"

    if technology == 'ONT':
        command = f"minimap2 -ax map-ont {reference_file} {fastqs} -t {threads}"
    elif technology == 'PB':
        command = f"minimap2 -ax map-pb {reference_file} {fastqs} -t {threads}"
    elif technology == 'HiFi':
        command = f"minimap2 -ax map-hifi {reference_file} {fastqs} -t {threads}"
    else:
        raise ValueError(f"Unknown technology: {technology}")

    command += f" | samtools view -bS -o {output_bam} -@ {threads}"
    subprocess.run(command, shell=True, check=True)

# Function to sort BAM file
def sort_bam(bam_file, threads):
    sorted_bam_file = bam_file.replace('.bam', '.sorted.bam')
    command = f"samtools sort {bam_file} -o {sorted_bam_file} -@ {threads}"
    print(f"Sorting BAM file with command: {command}")
    subprocess.run(command, shell=True, check=True)
    return sorted_bam_file

# Function to index BAM file
def index_bam(bam_file, threads):
    command = f"samtools index {bam_file} -@ {threads}"
    print(f"Indexing BAM file with command: {command}")
    subprocess.run(command, shell=True, check=True)

# Function to merge multiple BAM files
def merge_bams(bam_files, output_bam, threads):
    command = f"samtools merge -@ {threads} {output_bam} " + " ".join(bam_files)
    print(f"Merging BAM files with command: {command}")
    subprocess.run(command, shell=True, check=True)
    return output_bam

# Function to get multiple files
def find_fastq_files(output_dir, prefix):
    patterns = [
        f"{prefix}_*.fastq",
        f"{prefix}_*.fastq.gz",
        f"{prefix}_*.fq",
        f"{prefix}_*.fq.gz",
    ]

    files = []
    for pattern in patterns:
        files.extend(glob.glob(os.path.join(output_dir, pattern)))

    return sorted(files)

def filter_vcf_info(input_vcf, output_vcf):
    _open_in  = gzip.open if str(input_vcf).endswith('.gz')  else open
    _open_out = gzip.open if str(output_vcf).endswith('.gz') else open
    with _open_in(input_vcf, 'rt') as infile, _open_out(output_vcf, 'wt') as outfile:
        for line in infile:
            if line.startswith('#'):
                outfile.write(line)
                continue

            columns = line.strip().split('\t')
            info_field = columns[7]

            # Filter INFO fields to keep only INS_LEN or DEL_LEN
            info_parts = info_field.split(';')
            filtered_info = [item for item in info_parts if item.startswith('INS_LEN=') or item.startswith('DEL_LEN=')]

            columns[7] = ';'.join(filtered_info) if filtered_info else '.'
            outfile.write('\t'.join(columns) + '\n')

        print('VCF generated successfully')
