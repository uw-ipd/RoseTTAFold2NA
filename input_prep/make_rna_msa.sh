#!/bin/bash

# inputs
in_fasta="$1"
out_dir="$2"
out_tag="$3"

# resources
CPU="$4"
MEM="$5"

# databases
db0="$PIPEDIR/RNA/Rfam.cm";
db1="$PIPEDIR/RNA/rnacentral.fasta";
db2="$PIPEDIR/RNA/nt";
db0to1="$PIPEDIR/RNA/rfam_annotations.tsv.gz";
db0to2="$PIPEDIR/RNA/Rfam.full_region.gz";

max_aln_seqs=50000
max_target_seqs=50000
max_split_seqs=5000
max_hhfilter_seqs=5000
max_rfam_num=100

Lch=`grep -v '^>' $in_fasta | tr -d '\n' | wc -c`

mkdir -p $out_dir
cp $in_fasta $out_dir
cd $out_dir
in_fasta=`basename $in_fasta`

function retrieveSeq {
    tabfile=$1
    db=$2
    tag=$3

    head -n $max_aln_seqs $tabfile | awk '{if ($2<$3) print $1,($2-6)"-"($3+6),"plus"; else print $1,($3-6)"-"($2+6),"minus"}' > $tag.list
    split -l $max_split_seqs $tag.list $tag.list.split.

    for file in $tag.list.split.*
    do
        suffix=`echo $file | sed 's/.*\.list\.split\.//g'`
        blastdbcmd -db $db -entry_batch $tag.list.split.$suffix -out tmp.$tag.db.$suffix -outfmt ">Accession:%a_TaxID:%T @%s" &> /dev/null
        cat tmp.$tag.db.$suffix | tr '@' '\n' > $tag.db.$suffix
    done
    cat $tag.db.* | sed 's/_\([0-9]*\)_TaxID:0/_TaxID:\1/' > $tag.db  # fix for incorrect taxids
    rm $tag.db.* $tag.list.split.* tmp.$tag.db.* 
}

function a2mToFasta {
    infile=$1
    outfile=$2

    # a) run hhsearch reformat.pl
    $PIPEDIR/input_prep/reformat.pl -l 49999 sto fas $infile $infile.fas
    nhits=`grep '^>' $infile.fas | wc -l`

    # b) pad with gaps
    Laln=`head -n 2 $infile.fas | tail -1 | wc -c`
    gaptag=`for ((i=$Laln;i<=$Lch;i++)); do echo -n '-'; done`

    grep '^>' $in_fasta > $outfile
    grep -v '^>' $in_fasta | tr -d '\n' >> $outfile
    echo >> $outfile
    cat $infile.fas | awk -v tag="$gaptag" '{if (substr($0,1,1)==">") {print $0} else {print $0tag}}'>> $outfile
}

# cmscan on Rfam
echo "Run cmscan on Rfam"
cmscan --tblout cmscan.tblout -o cmscan.out --noali $db0 $in_fasta
families=`grep -v '^#' cmscan.tblout | head -n $max_rfam_num | uniq | awk '{print $2}' | sed -z 's/\n/|/g;s/|$/\n/'`
echo "Rfam families:" $families
rm cmscan.out cmscan.tblout

# Rfam->RNACentral
zcat $db0to1 | grep -E \'$families\' | awk '{print $1,1+$5,1+$6}' > rfam1.tab
head -n $max_aln_seqs rfam1.tab > rfam1.tab.tmp; mv rfam1.tab.tmp rfam1.tab
retrieveSeq rfam1.tab $db1 rfam1
rm rfam1.list rfam1.tab 

# Rfam->nt
zcat $db0to2 | grep -E \'$families\' | awk '{print $2,$3,$4}' > rfam2.tab
head -n $max_aln_seqs rfam2.tab > rfam2.tab.tmp; mv rfam2.tab.tmp rfam2.tab
retrieveSeq rfam2.tab $db2 rfam2
rm rfam2.list rfam2.tab 

if [[ -f "rfam1.db" || -f "rfam2.db" ]]
then
    cat rfam1.db rfam2.db > db0
    rm rfam1.db rfam2.db
fi

# blastn on RNACentral
echo "Run blastn on RNACentral"
blastn -num_threads $CPU -query $in_fasta -strand plus -db $db1 -out blastn1.tab -task blastn -max_target_seqs $max_target_seqs -outfmt '6 saccver sstart send evalue bitscore nident staxids'
retrieveSeq blastn1.tab $db1 blastn1
rm blastn1.list blastn1.tab 

# blastn on nt
echo "Run blastn on nt"
blastn -num_threads $CPU -query $in_fasta -strand both -db $db2 -out blastn2.tab -task blastn -max_target_seqs $max_target_seqs -outfmt '6 saccver sstart send evalue bitscore nident staxids'
retrieveSeq blastn2.tab $db2 blastn2
rm blastn2.list blastn2.tab 

# combine, remove redundant
echo "Cluster sequences"
throw_away_sequences=$(( $Lch*2/5 ));
cat db0 blastn*.db > trim.db
rm db0 blastn*.db
for cut in 1.00 0.99 0.95 0.90
do
    cd-hit-est-2d -T $CPU -i $in_fasta -i2 trim.db -c $cut -o cdhitest2d.db -l $throw_away_sequences -M 5000 &> /dev/null 
    cd-hit-est -T $CPU -i cdhitest2d.db -c $cut -o db -l $throw_away_sequences -M 5000 &> /dev/null 
    nhits=`grep '^>' db | wc -l`
    if [[ $nhits -lt $max_aln_seqs ]]
    then
        break
    fi
done
rm cdhitest2d.db cdhitest2d.db.clstr db.clstr

# nhmmer on previous hits
echo "Realign all with nhmmer"
for e_val in 1e-8 1e-7 1e-6 1e-3 1e-2 1e-1
do
    nhmmer --noali -A nhmmer.a2m --incE $e_val --cpu $CPU --watson $in_fasta db | grep 'no alignment saved'
    a2mToFasta nhmmer.a2m $out_tag.unfilter.afa
    hhfilter -i $out_tag.unfilter.afa -id 99 -cov 50 -o $out_tag.afa
    hitnum=`grep '^>' $out_tag.afa | wc -l`
    if [[  $hitnum -gt $max_hhfilter_seqs ]]
    then
        break
    fi
done

