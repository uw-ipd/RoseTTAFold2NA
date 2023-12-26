# RF2NA
GitHub repo for RoseTTAFold2 with nucleic acids

**New: April 13, 2023 v0.2**
* Updated weights (https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz) for better prediction of homodimer:DNA interactions and better DNA-specific sequence recognition
* Bugfixes in MSA generation pipeline
* Support for paired protein/RNA MSAs

## Installation

1. Clone the package
```
git clone https://github.com/uw-ipd/RoseTTAFold2NA.git
cd RoseTTAFold2NA
```

2. Create conda environment
All external dependencies are contained in `RF2na-linux.yml`
```
# create conda environment for RoseTTAFold2NA
conda env create -f RF2na-linux.yml
```
You also need to install NVIDIA's SE(3)-Transformer (**please use SE3Transformer in this repo to install**).
```
conda activate RF2NA
cd SE3Transformer
pip install --no-cache-dir -r requirements.txt
python setup.py install
cd ..
```

3. Download pre-trained weights under network directory
```
cd network
wget https://files.ipd.uw.edu/dimaio/RF2NA_apr23.tgz
tar xvfz RF2NA_apr23.tgz
ls weights/ # it should contain a 1.1GB weights file
cd ..
```

4. Download sequence and structure databases
```
# uniref30 [46G]
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
mkdir -p UniRef30_2020_06
tar xfz UniRef30_2020_06_hhsuite.tar.gz -C ./UniRef30_2020_06

# BFD [272G]
wget https://bfd.mmseqs.com/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz
mkdir -p bfd
tar xfz bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt.tar.gz -C ./bfd

# structure templates (including *_a3m.ffdata, *_a3m.ffindex)
wget https://files.ipd.uw.edu/pub/RoseTTAFold/pdb100_2021Mar03.tar.gz
tar xfz pdb100_2021Mar03.tar.gz

# RNA databases
mkdir -p RNA
cd RNA

# Rfam [300M]
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full_region.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz
gunzip Rfam.cm.gz
cmpress Rfam.cm

# RNAcentral [12G]
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/rfam/rfam_annotations.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/id_mapping/id_mapping.tsv.gz
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz
../input_prep/reprocess_rnac.pl id_mapping.tsv.gz rfam_annotations.tsv.gz   # ~8 minutes
gunzip -c rnacentral_species_specific_ids.fasta.gz | makeblastdb -in - -dbtype nucl  -parse_seqids -out rnacentral.fasta -title "RNACentral"

# nt [151G]
update_blastdb.pl --decompress nt
cd ..
```

## Usage
```
conda activate RF2NA
cd example
# run Protein/RNA prediction
../run_RF2NA.sh rna_pred rna_binding_protein.fa R:RNA.fa
# run Protein/DNA prediction
../run_RF2NA.sh dna_pred dna_binding_protein.fa D:DNA.fa
```
### Inputs
* The first argument to the script is the output folder
* The remaining arguments are fasta files for individual chains in the structure.  Use the tags `P:xxx.fa` `R:xxx.fa` `D:xxx.fa` `S:xxx.fa` to specify protein, RNA, double-stranded DNA, and single-stranded DNA, respectively.  Use the tag `PR:xxx.fa` to specify paired protein/RNA.    Each chain is a separate file; 'D' will automatically generate a complementary DNA strand to the input strand.  

### Expected outputs
* Outputs are written to the folder provided as the first argument (`dna_pred` and `rna_pred`).
* Model outputs are placed in a subfolder, `models` (e.g., `dna_pred.models`)
* You will get a predicted structre with estimated per-residue LDDT in the B-factor column (`models/model_00.pdb`)
* You will get a numpy `.npz` file (`models/model_00.npz`).  This can be read with `numpy.load` and contains three tables (L=complex length):
   - dist (L x L x 37) - the predicted distogram
   - lddt (L) - the per-residue predicted lddt
   - pae (L x L) - the per-residue pair predicted error
