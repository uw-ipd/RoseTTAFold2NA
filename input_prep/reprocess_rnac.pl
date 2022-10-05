#! /usr/bin/perl
use strict;

my $taxids = shift @ARGV;
my $idfile = shift @ARGV;

my %ids;
open(GZIN, "gunzip -c $taxids |") or die("gunzip $taxids: $!");
foreach my $line (<GZIN>) {
    my ($id,$taxid);
    ($id,$_,$_,$taxid,$_,$_) = split ' ',$line;
    if (not defined $ids{$id}) {
        $ids{$id} = []
    }
    if (not $taxid ~~ @{$ids{$id}}) {
        push (@{$ids{$id}}, $taxid)
    }
}
close(GZIN);

system ("mv $idfile $idfile.bak");
open (GZOUT, "| gzip -c > $idfile") or die("gzip $idfile: $!");
open(GZIN, "gunzip -c $idfile.bak |") or die("gunzip $idfile: $!");
foreach my $line (<GZIN>) {
    #URS0000000001	RF00177	109.4	3.3e-33	2	200	29	230	Bacterial small subunit ribosomal RNA
    my @fields = split /\t/,$line;
    my $id = $fields[0];

    foreach my $taxid (@{$ids{$id}}) {
        print GZOUT $id."_".$taxid."\t".join("\t",@fields[1..$#fields]);
    }
}
