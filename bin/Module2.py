#!/usr/bin/env python3
# SVModeller - Module 2

# Model probabilities, insertion features, and genome wide distribution to generate new events

# Input:
# - Genome-Wide Distribution (Genome_Wide_Distribution.tsv)
# - Insertion Features (Insertion_Features.tsv)
# - Event Probabilities or Number of each event to simulate (Probabilities.tsv)
# - OPTIONAL: Just in case of providing probabilities, total number of events to simulate (integer number)
# - Table with source loci to LINE-1 transductions (source_loci_LINE1.tsv)
# - Table with source loci to SVA transductions (source_loci_SVA.tsv)
# - Consensus sequences (consensus_sequences_complete.fa)
# - Reference genome (chm13v2.0.fa)
# - List of VNTR motifs (Separated_Motifs.tsv)
# - List of SVA VNTR motifs (SVA_VNTR_Motifs.txt)

# Output:
# - New insertion events sequences with their corresponding features (Insertions_table.tsv)
# - OPTIONAL: Variant Calling File (VCF) with insertion data

# Developers
# SVModeller has been developed by Ismael Vera-Munoz (orcid.org/0009-0009-2860-378X) at the Repetitive DNA Biology (REPBIO) Lab at the Centre for Genomic Regulation (CRG) (Barcelona 2024-2026)

# License
# SVModeller is distributed under the AGPL-3.0.

import argparse
import pandas as pd
import warnings
from functions import (
    set_seed,
    consensus_seqs,
    read_file_and_store_lines,
    probabilities_total_number,
    process_insertion_features_random_numbers,
    add_beg_end_columns,
    add_elements_columns,
    add_SVA_info,
    update_dataframe,
    add_source_gene_info,
    generate_insertion_seq,
    process_vntr_motifs,
    update_sequences,
    df_VCF,
    create_vcf_file
)

warnings.simplefilter(action='ignore', category=FutureWarning)

def main(consensus_path, probabilities_numbers_path, insertion_features_path, genome_wide_path, source_L1_path, source_SVA_path, motifs_path, SVA_VNTR_path, reference_fasta_path, chromosome_length_path, num_events, apply_VCF, seed):
    print(f'File with consensus sequences: {consensus_path}')
    print(f'File with probabilities or number of events: {probabilities_numbers_path}')
    print(f'File with insertions features: {insertion_features_path}')
    print(f'File with genome-wide distribution: {genome_wide_path}')
    print(f'File with source genes for L1 transductions: {source_L1_path}')
    print(f'File with source genes for SVA transductions: {source_SVA_path}')
    print(f'File with VNTR motifs: {motifs_path}')
    print(f'File with reference genome: {reference_fasta_path}')
    print(f'File with SVA VNTR motifs: {SVA_VNTR_path}')

    # Set seed
    set_seed(seed)
    # Get consensus sequences
    consensus_dict = consensus_seqs(consensus_path)
    # Open SVAs VNTR motifs file
    SVA_VNTR_motifs = read_file_and_store_lines(SVA_VNTR_path)
    # Get number or probabilities of events
    df_insertions1 = probabilities_total_number(probabilities_numbers_path, num_events)
    # Process insertion features and generate dict with random numbers based on distributions for each feature of every event
    dict_random = process_insertion_features_random_numbers(insertion_features_path, num_events, consensus_dict)
    # Add chromosome and start position
    df_insertions2 = add_beg_end_columns(df_insertions1, genome_wide_path)
    # Add the features of each event
    df_insertions3 = add_elements_columns(dict_random, df_insertions2)
    # Add SVA events additional details
    df_insertions4 = add_SVA_info(df_insertions3, consensus_dict)
    # Update df
    df_insertions5 = update_dataframe(df_insertions4, consensus_dict)
    # Add source gene information for transduction events
    df_insertions6 = add_source_gene_info(df_insertions5, source_L1_path, source_SVA_path)
    # Generate the insertion sequence and selected VNTR motifs
    df_insertions6['Sequence_Insertion'], df_insertions6['Selected_VNTR_Motifs'], df_insertions6['VNTR_Num_Motifs'] = zip(*df_insertions6.apply(lambda row: generate_insertion_seq(row, motifs_path, reference_fasta_path, consensus_dict, SVA_VNTR_motifs), axis=1))
    # Update VNTR motifs
    df_insertions6 = process_vntr_motifs(df_insertions6)
    # Update the insertion sequence
    df_insertions6 = update_sequences(df_insertions6)
    # Save the output
    df_insertions6.to_csv('Insertions_table.tsv', sep='\t', index=False)

    # If the VCF argument is provided, create a VCF file
    if apply_VCF:
        df_insertions7 = pd.read_csv('Insertions_table.tsv', sep='\t')
        # Process the df to transform it to VCF format
        df_VCF_format = df_VCF(df_insertions7, reference_fasta_path)
        # Create the VCF
        create_vcf_file(df_VCF_format, reference_fasta_path, chromosome_length_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate insertion sequences returned in tsv file.')
    parser.add_argument('--consensus_path', type=str, required=True, help='Path to file with consensus sequences.')
    parser.add_argument('--probabilities_numbers_path', type=str, required=True, help='Path to the TSV file probabilities or defined number of events.')
    parser.add_argument('--insertion_features_path', type=str, required=True, help='Path to the TSV file containing insertion features of events.')
    parser.add_argument('--genome_wide_path', type=str, required=True, help='Path to the TSV file containing genome-wide distribution of events.')
    parser.add_argument('--source_L1_path', type=str, required=True, help='Path to the TSV file containing loci for LINE-1 transductions.')
    parser.add_argument('--source_SVA_path', type=str, required=True, help='Path to the TSV file containing loci for SVA transductions.')
    parser.add_argument('--motifs_path', type=str, required=True, help='Path to the TSV file containing loci for SVA transductions.')
    parser.add_argument('--SVA_VNTR_path', type=str, required=True, help='Path to the TSV file containing loci for SVA transductions.')
    parser.add_argument('--reference_fasta_path', type=str, required=True, help='Path to file with reference genome.')
    parser.add_argument('--chromosome_length_path', type=str, required=True, help='Path to the chromosome length file.')
    parser.add_argument('--num_events', type=int, default=100, required=False, help='Number of events to sample (optional, just in case of providing probabilities).')
    parser.add_argument('--VCF', action='store_true', required=False, help='If specified, creates a Variant Calling File (VCF)')
    parser.add_argument('--seed', type=int, required=False, default=42, help='Random seed for reproducibility (default: 42).')

    args = parser.parse_args()

    main(args.consensus_path, args.probabilities_numbers_path, args.insertion_features_path, args.genome_wide_path, args.source_L1_path, args.source_SVA_path, args.motifs_path, args.SVA_VNTR_path, args.reference_fasta_path, args.chromosome_length_path, args.num_events, args.VCF, args.seed)
