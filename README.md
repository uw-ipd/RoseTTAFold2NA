# RF2NA
GitHub repo for RoseTTAFold2 with nucleic acids

## Installation

1. Clone the package
```
git clone https://github.com/uw-ipd/RoseTTAFold2NA.git
cd RoseTTAFold2NA
```

2. Create conda environment
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
```

3. Download pre-trained weights under network directory
```
cd network
wget https://files.ipd.uw.edu/dimaio/RF2NA_sep22.tgz
tar xvfz RF2NA_sep22.tgz
ls weights/ # it should contain a 1.6gb weights file
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

# Rfam [300M]
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.full_region.gz -C ./RNA
wget ftp://ftp.ebi.ac.uk/pub/databases/Rfam/CURRENT/Rfam.cm.gz -C ./RNA
gzip -d -f version.txt.gz

# RNAcentral [12G]
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/sequences/rnacentral_species_specific_ids.fasta.gz -C ./RNA
wget ftp://ftp.ebi.ac.uk/pub/databases/RNAcentral/current_release/rfam/rfam_annotations.tsv.gz -C ./RNA

# nt [151G]
cd RNA
update_blastdb.pl --decompress nt
cd ..
```

## Usage
```
conda activate RF2
cd example
../run_RF2.sh protein.fa R:rna.fa .
```
Use the tags 'P:xxx.fa' 'R:xxx.fa' 'D:xxx.fa' to specify protein, DNA, RNA respectively (default protein).  Each chain is a separate file (e.g., for double-stranded DNA, both strands need to be provided as separate fasta files).

## Expected outputs
You will get a prediction with estimated per-residue LDDT in the B-factor column (model.pdb)
