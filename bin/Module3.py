#!/usr/bin/env python3
# SVMoldeller - Module 3

# Process Deletions

# INPUT:
# - VCF with deletion data (VCF_Deletions.vcf)
# - Total number of events to simulate (integer number)
# - Chromosomes length (chr_length.txt)
# - OPTIONAL: Reference genome (chm13v2.0.fa) just if VCF file is desired
# - OPTIONAL: Window size for genome segmentation, by default 1 Mega base (integer number)

# OUTPUT:
# - Deletion regions (Deletions_table.tsv)
# - OPTIONAL: Variant Calling File (VCF) with deletion data

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
    classify_mutations_in_bins,
    normalize_columns,
    probabilities_df,
    generate_deletion_events,
    create_VCF
)

# Remove FutureWarnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def main(vcf_path, path_chromosome_length, num_events, bin_size, apply_VCF, reference_fasta_path, seed):
    # Print the paths of the input files
    print(f'VCF file with deletion data: {vcf_path}')
    print(f'Chromosome length file: {path_chromosome_length}')
    print(f'Number of events: {num_events}')
    print(f'Size of genomic bins (default: 1000000).: {bin_size}')

    # Set seed
    set_seed(seed)

    # Get data from VCF file
    table = read_vcf_file_BED(vcf_path, sv_type='deletion')

    # Process the data
    processed_table = process_bed_table(table, sv_type='deletion')

    # Obtain genome-wide distribution of the events & normalize it
    genome_wide_distribution = classify_mutations_in_bins(path_chromosome_length, bin_size, processed_table)
    normalize_columns(genome_wide_distribution)

    # Obtain probability of each event
    probabilities = probabilities_df(processed_table)

    # Generate the deletion events & save the output
    deletion_events = generate_deletion_events(probabilities, num_events, processed_table, genome_wide_distribution)
    deletion_events = deletion_events.rename(columns={'Event': 'name'})
    deletion_events.to_csv('Deletions_table.tsv', sep='\t', index=False)

    # If the VCF argument is provided, create a VCF file
    if apply_VCF:
        create_VCF(deletion_events, reference_fasta_path, path_chromosome_length)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process deletions from VCF to BED format.')
    parser.add_argument('--vcf_path', type=str, required=True, help='Path to the VCF file containing deletion data.')
    parser.add_argument('--path_chromosome_length', type=str, required=True, help='Path to the chromosome length file.')
    parser.add_argument('--num_events', type=int, required=True, help='Number of events to sample (mandatory).')
    parser.add_argument('--bin_size', type=int, default=1000000, required=False, help='Size of genomic bins (default: 1000000).')
    parser.add_argument('--VCF', action='store_true', required=False, help='If specified, creates a Variant Calling File (VCF)')
    parser.add_argument('--reference_fasta_path', type=str, required=False, help='Path to file with reference genome.')
    parser.add_argument('--seed', type=int, required=False, default=42, help='Random seed for reproducibility (default: 42).')

    args = parser.parse_args()
    # Check if --VCF is provided, and make sure all required arguments are there
    if args.VCF:
        if not args.reference_fasta_path:
            parser.print_help()
            raise ValueError("When --VCF is specified --reference_fasta_path is required.")

    main(args.vcf_path, args.path_chromosome_length, args.num_events, args.bin_size, args.VCF, args.reference_fasta_path, args.seed)
