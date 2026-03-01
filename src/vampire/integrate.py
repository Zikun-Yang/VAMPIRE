#! /usr/bin/env python3

# type hints
from typing import List, Dict, Tuple, Any, Optional
# basic packages for file operations, logging
import csv
import sys
import logging
import shutil
import subprocess
from tqdm import tqdm
from pathlib import Path
import polars as pl

logger = logging.getLogger(__name__)

"""
#
# codes for making necessary files for integration
#
"""
def make_minimap2_index(genome: str, alignment_params: str, alignment_index: str, threads: int) -> None:
    """
    Make minimap2 index for the genome.
    Inputs:
        genome : str, genome file path
        alignment_params : str, alignment parameters
        alignment_index : str, alignment index file path
    Outputs:
        None
    """
    subprocess.run(
        [
            "minimap2",
            *alignment_params.split(),
            "-t", str(threads),
            "-d", alignment_index,
            genome,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def make_minimap2_alignment(reference: str, query: str, alignment_params: str, alignment_file: str, threads: int) -> None:
    """
    Make minimap2 alignment for the reference and query.
    Inputs:
        reference : str, reference genome / index file path
        query : str, query genome file path
        alignment_params : str, alignment parameters
        alignment_file : str, alignment file path
    Outputs:
        None
    """
    subprocess.run(
        [
            "minimap2",
            *alignment_params.split(),
            "-t", str(threads),
            reference,
            query,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def make_chain_file(alignment_file: str, chain_file: str) -> None:
    """
    Make chain file for the reference and query.
    Inputs:
        alignment_file : str, PAF alignment file path
        chain_file : str, chain file path
    Outputs:
        None
    """
    subprocess.run(
        [
            "transanno",
            "minimap2chain",
            "--output", chain_file,
            alignment_file,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

def get_chain_direction(chain_file: str) -> None:
    """
    Get chain direction for the chain file.
    Inputs:
        chain_file : str, chain file path
    Outputs:
        None
    """
    chain_dir_filename = chain_file.replace(".chain", ".chain_direction.txt")
    subprocess.run(
        "grep",
        "chain",
        chain_file,
        ">",
        chain_dir_filename
    )

"""
#
# codes for xxxxxxxxxx
#
"""


"""
def extract_flanks(tr: TR, flank_len: int = 200) -> tuple[tuple[int, int], tuple[int, int]]:
    left = (max(0, tr.start - flank_len), tr.start)
    right = (tr.end, tr.end + flank_len)
    return left, right

def liftover_flank(
    chrom: str,
    start: int,
    end: int,
    chain_file: str
) -> FlankMapping:
    # 你这里可以是 subprocess 调 transanno
    # 我假设你最终能拿到这些信息
    try:
        result = call_transanno(chrom, start, end, chain_file)
        return FlankMapping(
            chrom=result.chrom,
            pos=(result.start + result.end) // 2,
            strand=result.strand,
            chain_id=result.chain_id,
            mapq=result.mapq,
            success=True
        )
    except Exception:
        return FlankMapping(None, None, None, None, 0, False)
        
"""

def anchor_tr(
    r_sample: str,
    q_sample: str,
    flank_len: int = 500
) -> None:
    """
    Anchor tandem repeats across samples.
    Inputs:
        r_sample : dict[str, Any], reference sample dictionary
        q_sample : dict[str, Any], query sample dictionary
        flank_len : int, flanking length
    Outputs:
        None
    """
    r_trs = r_sample["annotation"]
    q_trs = q_sample["annotation"]
    chain_file = f"{JOB_DIR}/chain/{r_sample}__{q_sample}.chain"

    # make coordinate bed file
    anchor_bed = f"{JOB_DIR}/anchor_tr.bed"
    with open(anchor_bed, "w") as f:
        for tr in q_trs.iter_rows():
            e1 = tr["start"]
            s1 = tr["start"] - min(flank_len, tr["end_flanking"])
            s2 = tr["end"]
            e2 = tr["end"] + min(flank_len, tr["end_flanking"])
            f.write(f"""{tr['chrom']}\t{s1}\t{e1}\t{tr['id']}-5p\n{
                        tr['chrom']}\t{s2}\t{e2}\t{tr['id']}-3p\n""")
    
    success_bed = f"{JOB_DIR}/success_tr.bed"
    fail_bed = f"{JOB_DIR}/fail_tr.bed"
    # run transanno
    subprocess.run(
        [
            "transanno",
            "liftbed",
            "-c", chain_file,
            "-o", success_bed,
            "-f", fail_bed,
            anchor_bed,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # read success and fail beds
    success_trs = pl.read_csv(success_bed, separator="\t", has_header=False)
    success_trs.columns = ["chrom", "start", "end", "id"] 
    success_trs = success_trs.with_columns(
        pl.col("id").str.split("-").str[0].astype(int).alias("locus_id")
    )
    print(success_trs)

    for locus_id in success_trs["locus_id"].unique():
        success_trs_locus = success_trs[success_trs["locus_id"] == locus_id]

        # flanking end cannot be lifted over
        if len(success_trs_locus) < 2:
            continue

        end5p = (
            success_trs_locus
            .filter(pl.col("id") == f"{locus_id}-5p")
            .row(0)
        )
        end3p = (
            success_trs_locus
            .filter(pl.col("id") == f"{locus_id}-3p")
            .row(0)
        )
        # different chromosomes
        if end5p[0] != end3p[0]:
            continue
        chrom = end5p[0]
        # different strands
        
        strand = end5p[3] # TODO
        if strand == '+':
            s = end5p[2]
            e = end3p[1]
        else:
            s = end3p[2]
            e = end5p[1]
        
        # intersect with trs
        ovlp_trs = r_trs.filter(
            (pl.col("chrom") == chrom) &
            (pl.col("start") >= e) &
            (pl.col("end") >= s)
        )
        match len(ovlp_trs):
            case 0:
                logger.warning(f"No tandem repeats found at {chrom}:{s}-{e}")
            case 1:
                # get r_tr locus_id
                r_tr_locus_id = ovlp_trs["id"].iloc[0]
                q_trs = q_trs.with_columns(
                    pl.when(pl.col("id") == locus_id)
                    .then(pl.lit(r_tr_locus_id))
                    .otherwise(pl.col("id"))
                    .alias("id")
                )
            case _:
                logger.warning(f"Multiple tandem repeats found at {chrom}:{s}-{e}")

        



"""
#
# main function for integrating tandem repeats across samples
#
"""
def run_integrate(cfg: dict[str, Any]) -> None:
    """
    Run the integrate function.
    Inputs:
        cfg : dict[str, Any], configuration dictionary
    Outputs:
        None
    """
    global JOB_DIR
    JOB_DIR = cfg["job_dir"]
    THREADS = cfg["threads"]
    ALIGNMENT_PARAMS = cfg["alignment_params"]
    samples = []

    # create job directory
    Path(cfg["prefix"]).mkdir(parents=True, exist_ok=True)
    Path(cfg["prefix"] + "/chain").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR).mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/alignment_index").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/alignment").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR + "/chain").mkdir(parents=True, exist_ok=True)

    # check software dependencies
    if not shutil.which("minimap2"):
        raise ValueError("minimap2 is not installed")
    if not shutil.which("transanno"):
        raise ValueError("transanno is not installed")

    # read input, format: sample, genome, annotation
    logger.info(f"Reading input file: {cfg['input']}")
    offset = 0
    with open(cfg["input"], "r") as f:
        reader = csv.reader(f, delimiter="\t", skipinitialspace=True)
        next(reader) # skip header
        for row in reader:
            sample, genome, annotation_file = row
            # read annotation
            anno = pl.read_csv(annotation_file, separator="\t", has_header=True)
            # add id and sample column
            n = len(anno)
            anno = anno.with_columns(
                pl.lit(sample).alias("sample"),
                pl.arange(offset, offset + n).alias("id")
            )
            offset += n
            # add flanking length column
            anno = anno.with_columns(
                (pl.col("start") - pl.col("end").shift(1)).alias("start_flanking"),
                (pl.col("start").shift(-1) - pl.col("end")).alias("end_flanking")
            )
            # fill nulls
            samples.append({
                "sample": sample,
                "genome": genome,
                "annotation": anno,
                "alignment_index": f"{JOB_DIR}/alignment_index/{sample}.mmi"
            })

    if cfg["reference"]:
        # reference genome mode
        logger.info("Used reference genome mode")
        sample_pairs = [(samples[0], sample) for sample in samples[1:]]
    else:
        # all-to-all mode (no reference genome)
        logger.info("Used all-to-all mode")
        sample_pairs = []
        for i in range(len(samples)):
            for j in range(i+1, len(samples)):
                sample_pairs.append((samples[i], samples[j]))

    logger.info(f"Found {len(sample_pairs)} sample pairs")
    
    # make alignment indexes
    logger.info("Starting making alignment indexes")
    if cfg["reference"]:
        for sample in tqdm(samples[:1], desc="Making indexes"):
            make_minimap2_index(sample["genome"], ALIGNMENT_PARAMS, sample["alignment_index"], THREADS)
    else:
        for sample in tqdm(samples[:-1], desc="Making indexes"):
            make_minimap2_index(sample["genome"], ALIGNMENT_PARAMS, sample["alignment_index"], THREADS)
    
    # make alignments
    logger.info("Starting making alignments")
    for sample_pair in tqdm(sample_pairs, desc="Making alignments"):
        alignment_file = f"{JOB_DIR}/alignment/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.paf"
        make_minimap2_alignment(sample_pair[0]["alignment_index"],
                                sample_pair[1]["genome"],
                                ALIGNMENT_PARAMS,
                                alignment_file,
                                THREADS)

    # make chain files
    logger.info("Starting making chain files")
    for sample_pair in tqdm(sample_pairs, desc="Making chains"):
        alignment_file = f"{JOB_DIR}/alignment/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.paf"
        chain_file = f"{JOB_DIR}/chain/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.chain"
        make_chain_file(alignment_file, chain_file)
        get_chain_direction(chain_file)

    # integrate tandem repeats
    offset = 0
    if cfg["reference"]:
        # reference genome mode
        for sample_pair in sample_pairs:
            anchor_tr(sample_pair[0], sample_pair[1], flank_len = cfg["flanking_length"])
            ####
            ####
            ####

    # keep chain files
    if cfg["keep_chain"]:
        logger.info("Keeping chain files")
        for sample_pair in sample_pairs:
            chain_file = f"{JOB_DIR}/chain/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.chain"
            shutil.copy2(chain_file, f"{cfg["prefix"]}/chain/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.chain")

    logger.info("Bye.")

    # copy log file
    shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg["prefix"]}.log")

    logging.shutdown()
    
    # remove temporary files
    if not cfg["debug"]:
        shutil.rmtree(JOB_DIR)