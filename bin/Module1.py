#!/usr/bin/env python3
# SVModeller - Module 1

# Obtain and process data from Variant Calling File (VCF)

# Input:
# - VCF with insertion data (VCF_Insertions.vcf)
# - Chromosomes length (chr_length.txt)
# - OPTIONAL: Window size for genome segmentation, by default 1 Mega base (integer number)

# Output:
# - Genome-wide distribution (Genome_Wide_Distribution.tsv)
# - Insertion features (Insertion_Features.tsv)
# - Event probabilities (Probabilities.tsv)
# - List of VNTR motifs (Separated_Motifs.tsv)
# - List of SVA VNTR motifs (SVA_VNTR_Motifs.txt)

# Developers
# SVModeller has been developed by Ismael Vera-Munoz (orcid.org/0009-0009-2860-378X) at the Repetitive DNA Biology (REPBIO) Lab at the Centre for Genomic Regulation (CRG) (Barcelona 2024-2026)

# License
# SVModeller is distributed under the AGPL-3.0.

import argparse
import warnings
from functions import (
    set_seed,
    read_vcf_file_BED,
    process_bed_table,
    create_dict,
    process_dictionary,
    insertion_features_df,
    genome_wide_distribution,
    probabilities_df
)

# Remove FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(file_path, chromosome_length, bin_size, seed):
    print(f'VCF file with insertions: {file_path}')
    print(f'File with chromosomes length: {chromosome_length}')
    print(f'Size of genomic bins (default: 1000000): {bin_size}')

    set_seed(seed)
    table = read_vcf_file_BED(file_path, sv_type='insertion')
    processed_table, vntr_df = process_bed_table(table, sv_type='insertion')
    dict1 = create_dict(processed_table)
    processed_dict = process_dictionary(dict1)
    insertion_features_df(processed_dict)
    genome_wide_distribution(chromosome_length, bin_size, processed_table)
    probabilities_df(processed_table, output_path='Probabilities.tsv')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Obtain and process data from Variant Calling File (VCF)')
    parser.add_argument('--file_path', type=str, required=True, help='Path to the VCF file containing insertion data.')
    parser.add_argument('--chromosome_length', type=str, required=True, help='Path to the chromosome length file.')
    parser.add_argument('--bin_size', type=int, required=False, default=1000000, help='Size of genomic bins (default: 1000000).')
    parser.add_argument('--seed', type=int, required=False, default=42, help='Random seed for reproducibility (default: 42).')

    args = parser.parse_args()
    main(args.file_path, args.chromosome_length, args.bin_size, args.seed)
