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
ls weights/ # it should contain a 800mb weights file
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
gunzip -c rnacentral_species_specific_ids.fasta.gz | makeblastdb -in - -dbtype nucl  -out rnacentral.fasta -title "RNACentral"

# nt [151G]
update_blastdb.pl --decompress nt
cd ..
```

## Usage
```
conda activate RF2NA
cd example
../run_RF2NA.sh t000_ protein.fa R:rna.fa
```
The first argument to the script is the output folder; remaining arguments are fasta files for individual chains in the structure.  Use the tags `P:xxx.fa` `R:xxx.fa` `D:xxx.fa` to specify protein, RNA, DNA respectively (default is protein).  Each chain is a separate file (e.g., for double-stranded DNA, both strands need to be provided as separate fasta files).  Outputs are written to the folder `t000_`.

## Expected outputs
You will get a prediction with estimated per-residue LDDT in the B-factor column (model_00.pdb)
