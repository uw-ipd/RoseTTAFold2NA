#!/bin/bash

# make the script stop when error (non-true exit code) is occured
set -e

############################################################
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('conda' 'shell.bash' 'hook' 2> /dev/null)"
eval "$__conda_setup"
unset __conda_setup
# <<< conda initialize <<<
############################################################

SCRIPT=`realpath -s $0`
export PIPEDIR=`dirname $SCRIPT`
HHDB="$PIPEDIR/pdb100_2021Mar03/pdb100_2021Mar03"

CPU="8"  # number of CPUs to use
MEM="64" # max memory (in GB)

WDIR=`realpath -s $1`  # working folder
mkdir -p $WDIR/log

conda activate RF2NA

# process protein (MSA + homology search)
function proteinMSA {
    seqfile=$1
    tag=$2

    ############################################################
    # 1. generate MSAs
    ############################################################
    if [ ! -s $WDIR/$tag.msa0.a3m ]
    then
        echo "Running HHblits"
        $PIPEDIR/input_prep/make_protein_msa.sh $seqfile $WDIR $tag $CPU $MEM > $WDIR/log/make_msa.$tag.stdout 2> $WDIR/log/make_msa.$tag.stderr
    fi


    ############################################################
    # 2. predict secondary structure for HHsearch run
    ############################################################
    if [ ! -s $WDIR/$tag.ss2 ]
    then
        echo "Running PSIPRED"
        $PIPEDIR/input_prep/make_ss.sh $WDIR/$tag.msa0.a3m $WDIR/$tag.ss2 > $WDIR/log/make_ss.$tag.stdout 2> $WDIR/log/make_ss.$tag.stderr
    fi


    ############################################################
    # 3. search for templates
    ############################################################
    if [ ! -s $WDIR/$tag.hhr ]
    then
        echo "Running hhsearch"
        HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $HHDB"
        cat $WDIR/$tag.ss2 $WDIR/$tag.msa0.a3m > $WDIR/$tag.msa0.ss2.a3m
        $HH -i $WDIR/$tag.msa0.ss2.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0 > $WDIR/log/hhsearch.$tag.stdout 2> $WDIR/log/hhsearch.$tag.stderr
    fi
}

# process RNA (MSA)
function RNAMSA {
    seqfile=$1
    tag=$2

    ############################################################
    # 1. generate MSAs
    ############################################################
    if [ ! -s $WDIR/$tag.afa ]
    then
        echo "Running rMSA (lite)"
        $PIPEDIR/input_prep/make_rna_msa.sh $seqfile $WDIR $tag $CPU $MEM > $WDIR/log/make_msa.$tag.stdout 2> $WDIR/log/make_msa.$tag.stderr
    fi
}

argstring=""

shift
for i in "$@"
do
    type=`echo $i | awk -F: '{if (NF==1) {print "P"} else {print $1}}'`
    type=${type^^}
    fasta=`echo $i | awk -F: '{if (NF==1) {print $1} else {print $2}}'`
    tag=`basename $fasta | sed -E 's/\.fasta$|\.fas$|\.fa$//'`

    if [ $type = 'P' ]
    then
        proteinMSA $fasta $tag
        argstring+="P:$WDIR/$tag.msa0.a3m:$WDIR/$tag.hhr:$WDIR/$tag.atab "
    elif [ $type = 'R' ]
    then
        RNAMSA $fasta $tag
        argstring+="R:$WDIR/$tag.afa "
    else [ $type = 'D' ]
        cp $fasta $WDIR/$tag.fa
        argstring+="D:$WDIR/$tag.fa "
    fi
done

############################################################
# 4. end-to-end prediction
############################################################
echo "Running RoseTTAFold2NA to predict structures"
mkdir -p $WDIR/models
python $PIPEDIR/network/predict.py \
    -inputs $argstring \
    -prefix $WDIR/models/model \
    -model $PIPEDIR/network/weights/RF2NA_sep22.pt \
    -db $HHDB #2> $WDIR/log/network.stderr #1> $WDIR/log/network.stdout 

echo "Done"
