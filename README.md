
# RandomBarcodes
This repository is for code accompanying the paper
### Fast trimer statistics facilitate accurate decoding of large random DNA barcode sets even at large sequencing error rates.
by William H. Press
(see paper abstract below)

### Usage
The file _Barcode_trigraphs_production.ipynb_ is a stand-alone Jupyter Notebook with sequential cells for these steps:
- Preliminaries: Define all functions to be used.
- Create and save a barcode library, or use an existing (saved) one.
- Make simulated barcode reads with specified error rates for substitutions, insertions, and deletions.
- For testing, move the code and simulated reads to GPU.
- Decode (find best barcode match to) the simulated reads.
- Using the known answers, compute precision and recall of the test.

An additional cell shows how to calculate the precision and recall for batch jobs with known answers.
The file _barcode_batch.py_ is used for batch production runs with large numbers (millions or more) of reads.
The file _barcode_allbatches.sh_ is a shell script showing how to use _barcode_batch.py_ to parallelize chunks of a single large file of reads across multiple concurrent processes and/or multiple GPUs. Two processes per GPU typically works well.

### Prerequisites
- Python (>= 3.8) and NumPy
- one or more GPUs
- PyTorch and its related CUDA tool chain

### Paper Abstract

Predefined sets of short DNA sequences are commonly used as barcodes to identify individual biomolecules in pooled populations. Such use requires either sufficiently small DNA error rates, or else an error-correction methodology. Most existing DNA error-correcting codes (ECCs) correct only one or two errors per barcode in sets of typically less than or the order of 10^4 barcodes. We here consider the use of random barcodes of sufficient length that they remain accurately decodable even with 6 or more errors and even at 10 or 20 percent nucleotide error rates. We show that length about 34 nt is sufficient even with as many as 10^6 or more barcodes. The obvious objection to this scheme is that it requires comparing every read to every possible barcode by a slow Levenshtein or Needleman-Wunsch comparison. We show that several orders of magnitude speedup can be achieved by (i) a fast triage method that compares only trimer (three consecutive nucleotide) occurence statistics, precomputed in linear time for both reads and barcodes, and (ii) the massive parallelism available on today's even commodity-grade GPUs. With 10^6 barcodes of length 34 and 10% DNA errors (substitutions and indels) we achieve in simulation 99.9% precision (decode accuracy) with 98.8% recall (read acceptance rate). Similarly high precision with somewhat smaller recall is achievable even with 20% DNA errors. The amortized computation cost on a commodity workstation with two GPUs (2022 capability and price) is estimated as between USD 0.15 and USD 0.60 per million decoded reads.
