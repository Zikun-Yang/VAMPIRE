import argparse
import logging
import time
import shutil
from pathlib import Path
from typing import List, Any, Dict

def _parse_int_list(s: str) -> List[int]:
    return [int(x) for x in s.split(",")]

def main():
    parser = argparse.ArgumentParser(
        prog='vampire',
        description='🧛 VAMPIRE: Comprehensive tool for annotating the motif variation and complex patterns in tandem repeats.'
    )

    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True

    # ------------------------------------------------------------
    # scan
    # ------------------------------------------------------------
    parser_scan = subparsers.add_parser('scan',
                                        description='VAMPIRE scan\n'
                                                    'Usage: vampire scan [options] <input.fa> <output_prefix>\n'
                                                    'For example: vampire scan <input.fa> <output_prefix>\n',
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        help='Scan and annotate tandem repeats on genome')
    # I/O Options
    file_group = parser_scan.add_argument_group('I/O Options')
    file_group.add_argument("input", type=str, help='Input FASTA file to scan TRs')
    file_group.add_argument("prefix", type=str, help="Output prefix")

    # General Options
    general_group = parser_scan.add_argument_group('General Options')
    general_group.add_argument("-t", "--thread", "--threads", dest="threads", type=int, default=8, help="Number of threads [8]")
    general_group.add_argument("--debug", action="store_true", help="Output debug info and keep temporary files [False]")
    general_group.add_argument("--seq-win-size", dest="seq_win_size", type=int, default=5000000, help="Window sequence size for scanning [5000000]")
    general_group.add_argument("--seq-ovlp-size", dest="seq_ovlp_size", type=int, default=100000, help="Overlap sequence size between windows [100000]")

    # Candidate Finding Options
    candidate_group = parser_scan.add_argument_group('Candidate Finding Options')
    candidate_group.add_argument("--ksize", type=_parse_int_list, default=[17, 13, 9, 5, 3], help="List of k-mer sizes for detect, e.g. --ksize 17,13,9,5,3 [17, 13, 9, 5, 3]")
    candidate_group.add_argument("--rolling-win-size", type=int, default=5, help="Rolling window size to compute smoothness score [5]")
    candidate_group.add_argument("--min-smoothness", type=int, default=50, help="Minimum smoothness score to call candidates [50]")

    # Alignment Options
    alignment_group = parser_scan.add_argument_group('Alignment Options')
    alignment_group.add_argument("--match-score", type=int, default=2, help="Match score for alignment [2]")
    alignment_group.add_argument("--mismatch-penalty", type=int, default=7, help="Mismatch penalty for alignment [7]")
    alignment_group.add_argument("--gap-open-penalty", type=int, default=7, help="Gap open penalty for alignment [7]")
    alignment_group.add_argument("--gap-extend-penalty", type=int, default=7, help="Gap extend penalty for alignment [7]")

    # Output Options
    output_group = parser_scan.add_argument_group('Output Options')
    output_group.add_argument("-p", "--max-period", dest="max_period", type=int, default=1000, help="Maximum period for output [1000]")
    output_group.add_argument("-s", "--min-score", dest="min_score", type=int, default=50, help="Minimum alignment score for output [50]")
    output_group.add_argument("-c", "--min-copy", dest="min_copy", type=float, default=1.5, help="Minimum copy number for output [1.5]")
    output_group.add_argument("--secondary", type=float, default=1.0, help="Minimum secondary annotation score compared with primary, set to 1 if no secondary annotation is needed [1.0]")
    output_group.add_argument("--format", type=str, choices=["brief", "trf", "bed"], default="trf", help="Output format [trf]") # TODO
    output_group.add_argument("--skip-cigar", action="store_true", default=False, help="Skip cigar string output [False]") # TODO
    output_group.add_argument("--skip-report", action="store_true", default=False, help="Skip HTML report generation [False]")

    # ------------------------------------------------------------
    # integrate
    # ------------------------------------------------------------
    parser_integrate = subparsers.add_parser('integrate',
                                        description='VAMPIRE integrate\n'
                                                    'Usage: vampire integrate [options] <input.tsv> <output_prefix>\n'
                                                    'For example: vampire integrate --reference <input.tsv> <output_prefix>\n',
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        help='Integrate tandem repeats across samples')
    
    # I/O Options
    file_group = parser_integrate.add_argument_group('I/O Options')
    file_group.add_argument('input', type=str, help='Input tsv file of sample, genome and annotation')
    file_group.add_argument('prefix', type=str, help='Output prefix')

    # General Options
    general_group = parser_integrate.add_argument_group('General Options')
    general_group.add_argument('-r', '--reference', action='store_true', help='Use the first sample as reference [False]')
    general_group.add_argument('--debug', action='store_true', help='Output debug info and keep temporary files [False]')
    general_group.add_argument('-t', '--thread', '--threads', dest="threads", type=int, default=8, help="Number of threads [16]")

    # Alignment Options
    alignment_group = parser_integrate.add_argument_group('Alignment Options')
    alignment_group.add_argument('-a', '--alignment-params', type=str, default="-x asm20 --secondary=no --cs", help="Alignment parameters for integration [-x asm20 --secondary=no --cs]")
    
    # Integration Options
    integration_group = parser_integrate.add_argument_group('Integration Options')
    integration_group.add_argument('-f', '--flanking-length', type=int, default=100, help="Flanking length for integration [100]")
    
    # Output Options
    output_group = parser_integrate.add_argument_group('Output Options')
    output_group.add_argument('--redo', action='store_true', help='Overwrite existing results [False]')

    # ------------------------------------------------------------
    # anno
    # ------------------------------------------------------------
    parser_anno = subparsers.add_parser('anno',
                                        description='VAMPIRE anno\n'
                                                    'Usage: vampire anno [options] <input.fa> <output_prefix>\n'
                                                    'For example: vampire anno --auto <input.fa> <output_prefix>\n'
                                                    '             vampire anno -k 13 --score 15 <CEN1.fa> <output_prefix>\n',
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        help='Annotate single tandem repeat locus')

    # I/O Options
    file_group = parser_anno.add_argument_group('I/O Options')
    file_group.add_argument('input', type=str, help='Input FASTA file to annotate')
    file_group.add_argument('prefix', type=str, help='Output prefix')

    # General Options
    general_group = parser_anno.add_argument_group('General Options')
    general_group.add_argument('-t', '--thread', '--threads', dest="threads", type=int, default=4, help='Number of threads [4]')
    general_group.add_argument('--auto', action='store_true', help='Automatically estimate k-mer size and related parameters [False]')
    general_group.add_argument('--debug', action='store_true', help='Output debug info and keep temporary files [False]')
    general_group.add_argument('--seq-win-size', type=int, default=5000, help='Parallel window size (bp) [5000]')
    general_group.add_argument('--seq-ovlp-size', type=int, default=1000, help='Overlap between consecutive windows (bp) [1000]')
    general_group.add_argument('-r', '--resource', type=int, default=50, help='Memory limit (GB) [50]')

    # Decomposition Options
    decompose_group = parser_anno.add_argument_group('Decomposition Options')
    decompose_group.add_argument('-k', '--ksize', type=int, default=9, help='k-mer size for building De Bruijn graph [9]')
    decompose_group.add_argument('-m', '--motif', type=str, default='base', help='Reference motif set path [base]')
    decompose_group.add_argument('-n', '--motifnum', type=int, default=30, help='Maximum number of motifs [30]')
    decompose_group.add_argument('--abud-threshold', type=float, default=0.01, help='Minimum threshold compared with top edge weight [0.01]')
    decompose_group.add_argument('--abud-min', type=int, default=3, help='Minimum edge weight in De Bruijn graph [3]')
    decompose_group.add_argument('--no-denovo', action='store_true', help='Do not de novo find motifs, use reference motifs to annotate [False]')

    # Annotation Options
    annotation_group = parser_anno.add_argument_group('Annotation Options')
    annotation_group.add_argument('-f', '--force', action='store_true', help='Add reference motifs into annotation module [False]')
    annotation_group.add_argument('--annotation-min-similarity', type=float, default=0.6, help='Min motif similarity to annotate [0.6]')
    annotation_group.add_argument('--finding-min-similarity', type=float, default=0.8, help='Min motif similarity to match a query in reference motif set [0.8]') # TODO: not used in anno.py
    annotation_group.add_argument("--match-score", type=int, default=2, help="Match score for alignment [2]")
    annotation_group.add_argument("--mismatch-penalty", type=int, default=7, help="Mismatch penalty for alignment [7]")
    annotation_group.add_argument("--gap-open-penalty", type=int, default=7, help="Gap open penalty for alignment [7]")
    annotation_group.add_argument("--gap-extend-penalty", type=int, default=7, help="Gap extend penalty for alignment [7]")

    # Output Options
    output_group = parser_anno.add_argument_group('Output Options')
    output_group.add_argument("--skip-report", action="store_true", default=False, help="Skip HTML report generation [False]")
    output_group.add_argument('-s', '--min-score', dest='min_score', type=float, default=5, help='Minimum row score for concise.tsv output [5]')

    # ------------------------------------------------------------
    # generator
    # ------------------------------------------------------------
    parser_generator = subparsers.add_parser('generator',
                                            description='VAMPIRE generator\n'
                                                        'Usage: vampire generator -m <motif> -l <length> -r <mutation_rate> -s <seed> -p <prefix>\n'
                                                        'For example: vampire generator -m GGC -l 1000 -r 0 -p <prefix>\n'
                                                        '             vampire generator -m GGC GGT -l 1000 -r 0 -p <prefix>\n',
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            help='Generate tandem repeat sequences from reference motifs')
    parser_generator.add_argument('-m', '--motifs', required=True, type=str, nargs='+', help='Input motif(s)')
    parser_generator.add_argument('-l', '--length', default=1000, type=int, help='Length of simulated tandem repeat')
    parser_generator.add_argument('-r', '--mutation-rate', default=0, type=float, help='Mutation rate, 0 - 1')
    parser_generator.add_argument('-s', '--seed', default=42, type=int, help='Random seed, DEFAULT: 42')
    parser_generator.add_argument('-p', '--prefix', required=True, type=str, help='Output prefix')
    parser_generator.add_argument('--debug', action='store_true', help='Output debug info [False]')

    # ------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------
    parser_evaluate = subparsers.add_parser('evaluate', 
                                            description='VAMPIRE evaluate\n'
                                                'Usage: vampire evaluate [options] [input_prefix] [output_prefix]\n'
                                                'For example: vampire evaluate [input_prefix] [output_prefix]\n',
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            help='Evaluate the tandem repeats.')
    parser_evaluate.add_argument('prefix', help='input prefix of raw results')
    parser_evaluate.add_argument('output', help='output prefix of evaluation results')
    parser_evaluate.add_argument('-t','--thread','--threads', dest="threads", type=int, default=6, help='thread number [6]')
    parser_evaluate.add_argument('-p','--percentage', type=int, default=75, help='threshold for identifying abnormal values (0-100) [75]')
    parser_evaluate.add_argument('-s','--show-distance', action='store_true', help='set to show detailed distance on heatmap')

    # ------------------------------------------------------------
    # refine
    # ------------------------------------------------------------
    parser_refine = subparsers.add_parser('refine', 
                                          description='VAMPIRE refine\n'
                                                'Usage: vampire refine [options] [prefix] [action]\n'
                                                'For example: vampire refine [prefix] [action]\n',
                                          formatter_class=argparse.RawTextHelpFormatter,
                                          help='Refine the tandem repeats.')
    parser_refine.add_argument("prefix", type=str, help="output prefix of raw results")
    parser_refine.add_argument("action", type=str, help="action file")
    parser_refine.add_argument("-o", "--out", type=str, default=None, help="output prefix of modified results [prefix.revised]")
    parser_refine.add_argument("-t", "--thread", "--threads", dest="threads", type=int, default=8, help="number of thread [8]")

    # ------------------------------------------------------------
    # identity
    # ------------------------------------------------------------
    parser_identity = subparsers.add_parser('identity', 
                                            description='VAMPIRE identity\n'
                                                'Usage: vampire identity [options] [input prefix] [output_prefix]\n'
                                                'For example: vampire identity [input prefix] [output_prefix]\n',
                                            formatter_class=argparse.RawTextHelpFormatter,
                                            help='Calculate the identity of the tandem repeats.')
    parser_identity.add_argument("prefix", type=str, help="prefix of the input file")
    parser_identity.add_argument("output", type=str, help="output prefix")
    parser_identity.add_argument("-w", "--window-size", type=int, default=100, help="window size (unit: motif)")
    parser_identity.add_argument("-t", "--thread", "--threads", dest="threads", type=int, default=30, help="thread number")
    parser_identity.add_argument("--mode", type=str, default='raw', help="mode: raw or invert")
    parser_identity.add_argument("--max-indel", type=int, default=0, help="maximum indel length")
    parser_identity.add_argument("--min-indel", type=int, default=0, help="minimum indel length")

    # get arguments
    args = parser.parse_args()
    cfg: dict[str, Any] = args.__dict__
    JOB_DIR = ".vampire/" + time.strftime("%Y%m%d_%H%M%S")
    cfg["job_dir"] = JOB_DIR

    # make directory for temporary files
    if Path(JOB_DIR).exists():
        shutil.rmtree(JOB_DIR)
    Path(JOB_DIR).mkdir(parents=True, exist_ok=True)

    # set up logging
    DEBUG = cfg.get("debug", False)
    handlers = [
        logging.StreamHandler(),                                     # terminal
        logging.FileHandler(JOB_DIR + "/log.log", encoding="utf-8")  # log file
    ]
    if DEBUG:
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
        logging.getLogger("numpy").setLevel(logging.WARNING)
        logging.getLogger("numba").setLevel(logging.WARNING)
        logging.getLogger("plotly").setLevel(logging.WARNING)
    else:
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=handlers
        )
    logger = logging.getLogger(__name__)
    logger.info(f"Created job directory: {JOB_DIR}")

    match cfg.get("command"):
        case "scan":
            from vampire.scan import run_scan
            run_scan(cfg)

        case "integrate":
            from vampire.integrate import run_integrate
            run_integrate(cfg)

        case "anno":
            from vampire.anno import run_anno
            run_anno(cfg)

        case "generator":
            from vampire.generator import run_generator
            run_generator(cfg)

        case "mkref":
            from vampire.mkref import run_mkref
            run_mkref(cfg)

        case "evaluate":
            from vampire.evaluate import run_evaluate
            run_evaluate(cfg)

        case "refine":
            from vampire.refine import run_refine
            run_refine(cfg)

        case "logo":
            from vampire.logo import run_logo
            run_logo(cfg)

        case "identity":
            from vampire.identity import run_identity
            run_identity(cfg)

        case _:
            parser.print_help()
            parser.exit(1)

    logging.shutdown()
    
    # remove temporary files
    if not cfg.get("debug", False):
        shutil.rmtree(JOB_DIR)

if __name__ == '__main__':
    main()
