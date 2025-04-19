#!/bin/bash
~/miniconda3/bin/activate trkscan
python revise_annotation.py -t 20 /home/zkyang/project/241205-trkscan/03-human_chr1_HOR/k13/human_chr1_active_HOR /home/zkyang/project/241205-trkscan/03-human_chr1_HOR/k13_revised/revision_action.tsv
