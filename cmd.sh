#!/bin/bash

# anno
vampire anno tests/001-anno_STR.fa tests/001-anno_STR

# generator
vampire generator -m GGC -l 1000 -r 0.01 -p tests/002-generator_reference
vampire generator -m GGC GGT -l 1000 -r 0.01 -p tests/002-generator_reference

# mkref
vampire mkref tests/003-mkref_data tests/003-mkref_reference.fa

# evaluate
vampire evaluate tests/001-anno_STR tests/004-evaluate

# refine
vampire refine tests/001-anno_STR tests/005-refine_action.tsv -o tests/005-anno_STR.revised

# logo
vampire logo tests/001-anno_STR tests/006-anno_STR_motif
vampire logo --type annotation tests/001-anno_STR tests/006-anno_STR_annotation

# identity
vampire identity -w 5 tests/001-anno_STR tests/007-anno_STR

#
