#!/bin/bash

# inputs
in_fasta="$1"
out_dir="$2"
tag="$3"

# resources
CPU="$4"
MEM="$5"

# sequence databases
if [ -z  "${DB_UR30}" ]; then
    DB_UR30="$PIPEDIR/UniRef30_2020_06/UniRef30_2020_06"
else
    DB_UR30=$DB_UR30
fi

if [ -z  "${DB_BFD}" ]; then
    DB_BFD="$PIPEDIR/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt"
else
    DB_BFD=$DB_BFD
fi

# setup hhblits command
HHBLITS_UR30="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4 -d $DB_UR30"
HHBLITS_BFD="hhblits -o /dev/null -mact 0.35 -maxfilt 100000000 -neffmax 20 -cov 25 -cpu $CPU -nodiff -realign_max 100000000 -maxseq 1000000 -maxmem $MEM -n 4 -d $DB_BFD"

mkdir -p $out_dir/hhblits
tmp_dir="$out_dir/hhblits"
out_prefix="$out_dir/$tag"

echo out_prefix $out_prefix

# perform iterative searches against UniRef30
prev_a3m="$in_fasta"
for e in 1e-10 1e-6 1e-3
do
    echo "Running HHblits against UniRef30 with E-value cutoff $e"
    $HHBLITS_UR30 -i $prev_a3m -oa3m $tmp_dir/t000_.$e.a3m -e $e -v 0
    hhfilter -id 90 -cov 75 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov75.a3m
    hhfilter -id 90 -cov 50 -i $tmp_dir/t000_.$e.a3m -o $tmp_dir/t000_.$e.id90cov50.a3m
    prev_a3m="$tmp_dir/t000_.$e.id90cov50.a3m"
    n75=`grep -c "^>" $tmp_dir/t000_.$e.id90cov75.a3m`
    n50=`grep -c "^>" $tmp_dir/t000_.$e.id90cov50.a3m`

    if ((n75>2000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov75.a3m ${out_prefix}.msa0.a3m
	    break
        fi
    elif ((n50>4000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.id90cov50.a3m ${out_prefix}.msa0.a3m
            break
        fi
    else
        continue
    fi
done

# perform iterative searches against BFD if it failes to get enough sequences
if [ ! -s ${out_prefix}.msa0.a3m ] 
then
    e=1e-3
    echo "Running HHblits against BFD with E-value cutoff $e"
    echo "hhblits command: $HHBLITS_BFD -i $prev_a3m -oa3m $tmp_dir/t000_.$e.bfd.a3m -e $e -v 0"
    $HHBLITS_BFD -i $prev_a3m -oa3m $tmp_dir/t000_.$e.bfd.a3m -e $e -v 0
    hhfilter -id 90 -cov 75 -i $tmp_dir/t000_.$e.bfd.a3m -o $tmp_dir/t000_.$e.bfd.id90cov75.a3m
    hhfilter -id 90 -cov 50 -i $tmp_dir/t000_.$e.bfd.a3m -o $tmp_dir/t000_.$e.bfd.id90cov50.a3m
    prev_a3m="$tmp_dir/t000_.$e.bfd.id90cov50.a3m"
    n75=`grep -c "^>" $tmp_dir/t000_.$e.bfd.id90cov75.a3m`
    n50=`grep -c "^>" $tmp_dir/t000_.$e.bfd.id90cov50.a3m`

    if ((n75>2000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.bfd.id90cov75.a3m ${out_prefix}.msa0.a3m
        fi
    elif ((n50>4000))
    then
        if [ ! -s ${out_prefix}.msa0.a3m ]
        then
            cp $tmp_dir/t000_.$e.bfd.id90cov50.a3m ${out_prefix}.msa0.a3m
        fi
    fi
fi

if [ ! -s ${out_prefix}.msa0.a3m ]
then
    cp $prev_a3m ${out_prefix}.msa0.a3m
fi
