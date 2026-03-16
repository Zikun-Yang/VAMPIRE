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
    result = subprocess.run(
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
    if result.returncode != 0:
        raise RuntimeError(f"minimap2 index creation failed for {genome}")

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
    with open(alignment_file, "w") as fo:
        result = subprocess.run(
            [
                "minimap2",
                *alignment_params.split(),
                "-t", str(threads),
                reference,
                query,
            ],
            stdout=fo,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            raise RuntimeError(f"minimap2 alignment failed for {reference} vs {query}")

def make_chain_file(alignment_file: str, chain_file: str) -> None:
    """
    Make chain file for the reference and query.
    Inputs:
        alignment_file : str, PAF alignment file path
        chain_file : str, chain file path
    Outputs:
        None
    """
    result = subprocess.run(
        [
            "transanno",
            "minimap2chain",
            "--output", chain_file,
            alignment_file,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Chain file creation failed for {alignment_file}")

def get_chain_direction(chain_file: str) -> None:
    """
    Get chain direction for the chain file.
    Inputs:
        chain_file : str, chain file path
    Outputs:
        None
    """
    chain_dir_filename = chain_file.replace(".chain", ".chain_direction.txt")
    with open(chain_dir_filename, "w") as f:
        result = subprocess.run(
            ["grep", "chain", chain_file],
            stdout=f,
            stderr=subprocess.DEVNULL,
        )
        if result.returncode != 0:
            logger.warning(f"Failed to extract chain direction from {chain_file}")

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
    r_sample: Dict[str, Any],
    q_sample: Dict[str, Any],
    flank_len: int = 500
) -> pl.DataFrame:
    """
    Anchor tandem repeats across samples.
    Inputs:
        r_sample : dict[str, Any], reference sample dictionary
        q_sample : dict[str, Any], query sample dictionary
        flank_len : int, flanking length
    Outputs:
        pl.DataFrame: Updated query annotation with anchored IDs
    """
    r_trs = r_sample["annotation"]
    q_trs = q_sample["annotation"]
    chain_file = f"{JOB_DIR}/chain/{r_sample['sample']}__{q_sample['sample']}.chain"

    # make coordinate bed file
    anchor_bed = f"{JOB_DIR}/anchor_tr_{q_sample['sample']}.bed"
    with open(anchor_bed, "w") as f:
        for tr in q_trs.iter_rows(named=True):
            e1 = tr["start"]
            # Use start_flanking for 5' end, end_flanking for 3' end
            # Handle missing flanking values (use infinity as default, then clamp to flank_len)
            start_flank = tr.get("start_flanking")
            end_flank = tr.get("end_flanking")
            if start_flank is None or start_flank == float('inf') or start_flank > flank_len:
                start_flank = flank_len
            if end_flank is None or end_flank == float('inf') or end_flank > flank_len:
                end_flank = flank_len
            s1 = tr["start"] - min(flank_len, start_flank)
            s2 = tr["end"]
            e2 = tr["end"] + min(flank_len, end_flank)
            f.write(f"{tr['chrom']}\t{s1}\t{e1}\t{tr['id']}-5p\n")
            f.write(f"{tr['chrom']}\t{s2}\t{e2}\t{tr['id']}-3p\n")
    
    success_bed = f"{JOB_DIR}/success_tr_{q_sample['sample']}.bed"
    fail_bed = f"{JOB_DIR}/fail_tr_{q_sample['sample']}.bed"
    # run transanno
    result = subprocess.run(
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
    if result.returncode != 0:
        logger.warning(f"transanno liftbed failed for {r_sample['sample']} vs {q_sample['sample']}")
        return q_trs

    # read success and fail beds
    if not Path(success_bed).exists() or Path(success_bed).stat().st_size == 0:
        logger.warning(f"No successful liftover for {r_sample['sample']} vs {q_sample['sample']}")
        return q_trs
    
    success_trs = pl.read_csv(success_bed, separator="\t", has_header=False)
    if len(success_trs) == 0:
        logger.warning(f"Empty success bed file for {r_sample['sample']} vs {q_sample['sample']}")
        return q_trs
    
    success_trs.columns = ["chrom", "start", "end", "id"] 
    success_trs = success_trs.with_columns(
        pl.col("id").str.split("-").str[0].cast(pl.Int64).alias("locus_id")
    )
    logger.debug(f"Successfully lifted over {len(success_trs)} flanking regions for {q_sample['sample']}")

    # Read chain direction file to get strand information
    chain_dir_file = chain_file.replace(".chain", ".chain_direction.txt")
    strand_map = {}
    if Path(chain_dir_file).exists():
        with open(chain_dir_file, "r") as f:
            for line in f:
                if line.startswith("chain"):
                    parts = line.strip().split()
                    if len(parts) >= 6:
                        # chain format: chain score tName tSize tStrand tStart ...
                        strand_map[parts[2]] = parts[4]  # chromosome -> strand
    
    # Create a mapping from query locus_id to reference locus_id
    id_mapping = {}
    
    for locus_id in success_trs["locus_id"].unique():
        success_trs_locus = success_trs[success_trs["locus_id"] == locus_id]

        # flanking end cannot be lifted over
        if len(success_trs_locus) < 2:
            continue

        end5p_row = success_trs_locus.filter(pl.col("id") == f"{locus_id}-5p")
        end3p_row = success_trs_locus.filter(pl.col("id") == f"{locus_id}-3p")
        
        if len(end5p_row) == 0 or len(end3p_row) == 0:
            continue
            
        end5p = end5p_row.row(0)
        end3p = end3p_row.row(0)
        
        # different chromosomes
        if end5p[0] != end3p[0]:
            continue
        chrom = end5p[0]
        
        # Get strand from chain direction file or assume positive
        strand = strand_map.get(chrom, '+')
        
        # Calculate lifted over TR region
        # For positive strand: 5p end -> start, 3p start -> end
        # For negative strand: 3p end -> start, 5p start -> end
        if strand == '+':
            s = end5p[2]  # 5p end becomes start
            e = end3p[1]  # 3p start becomes end
        else:
            s = end3p[2]  # 3p end becomes start
            e = end5p[1]  # 5p start becomes end
        
        # Fix: correct interval intersection logic
        # Two intervals [s1, e1] and [s2, e2] overlap if: s1 <= e2 and s2 <= e1
        ovlp_trs = r_trs.filter(
            (pl.col("chrom") == chrom) &
            (pl.col("start") <= e) &
            (pl.col("end") >= s)
        )
        
        match len(ovlp_trs):
            case 0:
                logger.debug(f"No tandem repeats found at {chrom}:{s}-{e} for locus {locus_id}")
            case 1:
                # get r_tr locus_id
                r_tr_locus_id = ovlp_trs["id"][0]
                id_mapping[locus_id] = r_tr_locus_id
                logger.debug(f"Anchored locus {locus_id} to reference locus {r_tr_locus_id} at {chrom}:{s}-{e}")
            case _:
                # If multiple overlaps, choose the one with best overlap
                # Calculate overlap for each
                best_overlap = 0
                best_id = None
                for r_tr in ovlp_trs.iter_rows(named=True):
                    overlap = min(e, r_tr["end"]) - max(s, r_tr["start"])
                    if overlap > best_overlap:
                        best_overlap = overlap
                        best_id = r_tr["id"]
                if best_id is not None:
                    id_mapping[locus_id] = best_id
                    logger.debug(f"Anchored locus {locus_id} to reference locus {best_id} (best overlap) at {chrom}:{s}-{e}")
                else:
                    logger.warning(f"Multiple tandem repeats found at {chrom}:{s}-{e} for locus {locus_id}, but no valid overlap")
    
    # Apply ID mapping to q_trs
    if id_mapping:
        # Create a mapping expression
        q_trs = q_trs.with_columns(
            pl.col("id").map_elements(
                lambda x: id_mapping.get(x, x),
                return_dtype=pl.Int64
            ).alias("id")
        )
        logger.info(f"Anchored {len(id_mapping)} TRs from {q_sample['sample']} to {r_sample['sample']}")
    else:
        logger.info(f"No TRs anchored from {q_sample['sample']} to {r_sample['sample']}")
    
    return q_trs

        



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
    global JOB_DIR, PREFIX
    JOB_DIR = cfg["job_dir"]
    PREFIX = cfg["prefix"]
    THREADS = cfg["threads"]
    ALIGNMENT_PARAMS = cfg["alignment_params"]
    samples = []

    # create job directory
    Path(PREFIX).mkdir(parents=True, exist_ok=True)
    Path(PREFIX + "/alignment_index").mkdir(parents=True, exist_ok=True)
    Path(PREFIX + "/alignment").mkdir(parents=True, exist_ok=True)
    Path(PREFIX + "/chain").mkdir(parents=True, exist_ok=True)
    Path(JOB_DIR).mkdir(parents=True, exist_ok=True)

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
            if sample.startswith("#"):
                logger.info(f"Skipping {sample.replace("#", "")}")
                continue
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
            # Calculate flanking regions within each chromosome
            # Sort by chrom and start to ensure proper ordering
            if "chrom" in anno.columns:
                anno = anno.sort(["chrom", "start"])
                anno = anno.with_columns(
                    pl.col("end").shift(1).over("chrom").alias("prev_end"),
                    pl.col("start").shift(-1).over("chrom").alias("next_start")
                )
            else:
                # If no chrom column, calculate globally
                anno = anno.sort("start")
                anno = anno.with_columns(
                    pl.col("end").shift(1).alias("prev_end"),
                    pl.col("start").shift(-1).alias("next_start")
                )
            anno = anno.with_columns(
                (pl.col("start") - pl.col("prev_end")).alias("start_flanking"),
                (pl.col("next_start") - pl.col("end")).alias("end_flanking")
            )
            # Fill nulls with infinity to indicate no flanking region
            anno = anno.with_columns(
                pl.col("start_flanking").fill_null(float('inf')),
                pl.col("end_flanking").fill_null(float('inf'))
            )
            # Drop temporary columns
            anno = anno.drop(["prev_end", "next_start"])
            samples.append({
                "sample": sample,
                "genome": genome,
                "annotation": anno,
                "alignment_index": f"{PREFIX}/alignment_index/{sample}.mmi"
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
            if cfg["redo"] or not Path(sample["alignment_index"]).exists():
                make_minimap2_index(sample["genome"], ALIGNMENT_PARAMS, sample["alignment_index"], THREADS)
    else:
        for sample in tqdm(samples[:-1], desc="Making indexes"):
            if cfg["redo"] or not Path(sample["alignment_index"]).exists():
                make_minimap2_index(sample["genome"], ALIGNMENT_PARAMS, sample["alignment_index"], THREADS)
    
    # make alignments
    logger.info("Starting making alignments")
    for sample_pair in tqdm(sample_pairs, desc="Making alignments"):
        alignment_file = f"{PREFIX}/alignment/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.paf"
        if cfg["redo"] or not Path(alignment_file).exists():
            make_minimap2_alignment(sample_pair[0]["alignment_index"],
                                    sample_pair[1]["genome"],
                                    ALIGNMENT_PARAMS,
                                    alignment_file,
                                    THREADS)

    # make chain files
    logger.info("Starting making chain files")
    for sample_pair in tqdm(sample_pairs, desc="Making chains"):
        alignment_file = f"{PREFIX}/alignment/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.paf"
        chain_file = f"{PREFIX}/chain/{sample_pair[0]['sample']}__{sample_pair[1]['sample']}.chain"
        if cfg["redo"] or not Path(chain_file).exists():
            make_chain_file(alignment_file, chain_file)
            get_chain_direction(chain_file)

    # integrate tandem repeats
    logger.info("Starting TR integration")
    flank_len = cfg.get("flanking_length", 100)
    
    if cfg["reference"]:
        # reference genome mode: anchor all query samples to reference
        reference_sample = samples[0]
        for sample_pair in tqdm(sample_pairs, desc="Anchoring TRs"):
            query_sample = sample_pair[1]
            logger.info(f"Anchoring {query_sample['sample']} to {reference_sample['sample']}")
            updated_anno = anchor_tr(reference_sample, query_sample, flank_len=flank_len)
            # Update the annotation in the sample dict
            query_sample["annotation"] = updated_anno
    else:
        # all-to-all mode: need to build a graph of matches and resolve conflicts
        logger.info("All-to-all mode: building TR anchor graph")
        # For simplicity, we'll anchor each pair independently
        # In a more sophisticated implementation, we'd use a graph algorithm
        # to resolve multi-way matches
        # Create a mapping from sample name to sample dict for easy lookup
        sample_dict = {s["sample"]: s for s in samples}
        for sample_pair in tqdm(sample_pairs, desc="Anchoring TRs"):
            r_sample_name = sample_pair[0]["sample"]
            q_sample_name = sample_pair[1]["sample"]
            logger.info(f"Anchoring {q_sample_name} to {r_sample_name}")
            updated_anno = anchor_tr(sample_pair[0], sample_pair[1], flank_len=flank_len)
            # Update the annotation in the original samples list
            sample_dict[q_sample_name]["annotation"] = updated_anno
    
    # Merge all annotations
    logger.info("Merging annotations from all samples")
    all_annotations = []
    for sample in samples:
        anno = sample["annotation"]
        all_annotations.append(anno)
    
    merged_anno = pl.concat(all_annotations)
    
    # Write output files
    output_file = f"{PREFIX}/integrated_trs.tsv"
    logger.info(f"Writing integrated TRs to {output_file}")
    merged_anno.write_csv(output_file, separator="\t")
    
    # Generate statistics
    total_trs = len(merged_anno)
    unique_ids = merged_anno["id"].n_unique()
    samples_count = merged_anno["sample"].n_unique()
    
    logger.info(f"Integration complete:")
    logger.info(f"  Total TRs: {total_trs}")
    logger.info(f"  Unique anchored IDs: {unique_ids}")
    logger.info(f"  Samples: {samples_count}")
    
    # Write statistics file
    stats_file = f"{PREFIX}/integration_stats.txt"
    with open(stats_file, "w") as f:
        f.write(f"Total TRs: {total_trs}\n")
        f.write(f"Unique anchored IDs: {unique_ids}\n")
        f.write(f"Number of samples: {samples_count}\n")
        f.write(f"Average TRs per sample: {total_trs / samples_count:.2f}\n")

    logger.info("Bye.")

    # copy log file
    shutil.copy2(f"{JOB_DIR}/log.log", f"{cfg["prefix"]}.log")