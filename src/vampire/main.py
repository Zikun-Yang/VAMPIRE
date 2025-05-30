import argparse

from vampire.anno import run_anno
from vampire.generator import run_generator
from vampire.mkref import run_mkref
from vampire.evaluate import run_evaluate
from vampire.refine import run_refine
from vampire.logo import run_logo
from vampire.identity import run_identity

def main():
    parser = argparse.ArgumentParser(
        prog='vampire',
        description='🧛VAMPIRE: An integrated tool for annotating the motif variation and complex patterns in tandem repeats.'
    )

    subparsers = parser.add_subparsers(title='subcommands', dest='command')
    subparsers.required = True

    # ------------------------------------------------------------
    # anno
    # ------------------------------------------------------------
    # parser_merge = subparsers.add_parser('merge', help='Merge multiple files')
    # parser_merge.add_argument('--input', required=True, help='Input directory')
    # parser_merge.add_argument('--output', required=True, help='Output file')
    # parser_merge.set_defaults(func=run_merge)

    # ------------------------------------------------------------
    # anno
    # ------------------------------------------------------------
    parser_anno = subparsers.add_parser('anno',
                                        description='VAMPIRE anno v0.3.0\n'
                                                    'Usage: vampire anno [--auto] [options] [input.fa] [output_prefix]\n'
                                                    'For example: vampire anno --auto [input.fa] [output_prefix]\n'
                                                    '             vampire anno -k 13 -s 15 [CEN1.fa] [output_prefix]\n',
                                        formatter_class=argparse.RawTextHelpFormatter,
                                        help='annotate tandem repeat sequences')

    # I/O Options
    file_group = parser_anno.add_argument_group('I/O Options')
    file_group.add_argument('input', help='Input FASTA file you want to annotate')
    file_group.add_argument('prefix', help='Output prefix')

    # General Options
    general_group = parser_anno.add_argument_group('General Options')
    general_group.add_argument('-t', '--thread', type=int, default=1, help='Number of threads [1]')
    general_group.add_argument('--AUTO', action='store_true', help='Automatically estimate parameters [False]')
    general_group.add_argument('--debug', action='store_true', help='Output running time of each module [False]')
    general_group.add_argument('--window-length', type=int, default=5000, help='Parallel window size [5000]')
    general_group.add_argument('--overlap-length', type=int, default=1000, help='Windows overlap size [1000]')
    general_group.add_argument('-r', '--resource', type=int, default=50, help='Memory limit (GB) [50]')

    # Decomposition Options
    decompose_group = parser_anno.add_argument_group('Decomposition Options')
    decompose_group.add_argument('-k', '--ksize', type=int, default=9, help='k-mer size for building De Bruijn graph [9]')
    decompose_group.add_argument('-m', '--motif', type=str, default='base', help='Reference motif set path [base]')
    decompose_group.add_argument('-n', '--motifnum', type=int, default=30, help='Maximum number of motifs [30]')
    decompose_group.add_argument('--abud-threshold', type=float, default=0.01, help='Minimum threshold compared with top edge weight [0.01]')
    decompose_group.add_argument('--abud-min', type=int, default=3, help='Minimum edge weight in De Bruijn graph [3]')
    decompose_group.add_argument('--plot', action='store_true', help='Paint De Bruijn graph for each window [False]')
    decompose_group.add_argument('--no-denovo', action='store_true', help='Do not de novo find motifs, use reference motifs to annotate [False]')

    # Annotation Options
    annotation_group = parser_anno.add_argument_group('Annotation Options')
    annotation_group.add_argument('-f', '--force', action='store_true', help='Add reference motifs into annotation [False]')
    annotation_group.add_argument('--annotation-dist-ratio', type=float, default=0.4, help='Max distance to map = 0.4 * motif length [0.4]')
    annotation_group.add_argument('--finding-dist-ratio', type=float, default=0.2, help='Max distance to query in reference motif set = 0.2 * motif length [0.2]')
    annotation_group.add_argument('--match-score', type=float, default=1, help='Score per matched base [1]')
    annotation_group.add_argument('--lendif-penalty', type=float, default=0.01, help='Penalty for length difference [0.01]')
    annotation_group.add_argument('--gap-penalty', type=float, default=1, help='Penalty per skipped base [1]')
    annotation_group.add_argument('--distance_penalty', type=float, default=1.5, help='Penalty per distance [1.5]')
    annotation_group.add_argument('--perfect-bonus', type=float, default=0.5, help='Bonus for perfect match [0.5]')

    # Output Options
    output_group = parser_anno.add_argument_group('Output Options')
    output_group.add_argument('--quiet', action='store_true', help="Don't output thread completion info")
    output_group.add_argument('-s', '--score', type=float, default=5, help='Minimum output score [5]')

    parser_anno.set_defaults(func=run_anno, _parser=parser_anno)

    # ------------------------------------------------------------
    # generator
    # ------------------------------------------------------------
    parser_generator = subparsers.add_parser('generator',
                                            description='VAMPIRE generator v0.1.0\n'
                                                'Usage: vampire generator -m [motif] -l [length] -r [mutation_rate] -s [seed] -p [output_prefix]\n'
                                                'For example: vampire generator -m "GGC" -l 1000 -r 0 -p [output_prefix]\n',
                                             help='Generate the reference database from annotation')
    parser_generator.add_argument('-m', '--motifs', required=True, type=str, nargs='+', help='Input motif(s)')
    parser_generator.add_argument('-l', '--length', default=1000, type=int, help='Length of simulated tandem repeat')
    parser_generator.add_argument('-r', '--mutation-rate', default=0, type=float, help='Mutation rate, 0 - 1')
    parser_generator.add_argument('-s', '--seed', default=42, type=int, help='Random seed, DEFAULT: 42')
    parser_generator.add_argument('-p', '--prefix', required=True, type=str, help='Output prefix')
    parser_generator.set_defaults(func=run_generator)

    # ------------------------------------------------------------
    # mkref
    # ------------------------------------------------------------
    parser_mkref = subparsers.add_parser('mkref', 
                                         description='VAMPIRE mkref v0.1.0\n'
                                            'Usage: vampire mkref [options] [prefix] [output_prefix]\n'
                                            'For example: vampire mkref [prefix] [output_prefix]\n',
                                         help='Make the reference database from annotation result')
    parser_mkref.add_argument('prefix', type=str, help='annotation result prefix')
    parser_mkref.add_argument('output', type=str, help='output')
    parser_mkref.set_defaults(func=run_mkref)

    # ------------------------------------------------------------
    # evaluate
    # ------------------------------------------------------------
    parser_evaluate = subparsers.add_parser('evaluate', 
                                            description='VAMPIRE evaluate v0.1.0\n'
                                                'Usage: vampire evaluate [options] [input_prefix] [output_prefix]\n'
                                                'For example: vampire evaluate [input_prefix] [output_prefix]\n',
                                            help='Evaluate the tandem repeats.')
    parser_evaluate.add_argument('prefix', help='input prefix of raw results')
    parser_evaluate.add_argument('output', help='output prefix of evaluation results')
    parser_evaluate.add_argument('-t','--thread', type=int, default=6, help='thread number [6]')
    parser_evaluate.add_argument('-p','--percentage', type=int, default=75, help='threshold for identifying abnormal values (0-100) [75]')
    parser_evaluate.add_argument('-s','--show-distance', action='store_true', help='set to show detailed distance on heatmap')
    parser_evaluate.set_defaults(func=run_evaluate)

    # ------------------------------------------------------------
    # refine
    # ------------------------------------------------------------
    parser_refine = subparsers.add_parser('refine', 
                                          description='VAMPIRE refine v0.1.0\n'
                                                'Usage: vampire refine [options] [prefix] [action]\n'
                                                'For example: vampire refine [prefix] [action]\n',
                                          help='Refine the tandem repeats.')
    parser_refine.add_argument("prefix", type=str, help="output prefix of raw results")
    parser_refine.add_argument("action", type=str, help="action file")
    parser_refine.add_argument("-o", "--out", type=str, default=None, help="output prefix of modified results [prefix.revised]")
    parser_refine.add_argument("-t", "--thread", type=int, default=8, help="number of thread [8]")
    parser_refine.set_defaults(func=run_refine)

    # ------------------------------------------------------------
    # logo
    # ------------------------------------------------------------
    parser_logo = subparsers.add_parser('logo', 
                                        description='VAMPIRE logo v0.1.0\n'
                                            'Usage: vampire logo [options] [input prefix] [outputprefix]\n'
                                            'For example: vampire logo [input prefix] [output_prefix]\n',
                                        help='Generate the logo of the tandem repeats.')
    parser_logo.add_argument("prefix", type=str,  help="prefix\nfor motif file, plot seq Logo of reference motifs\nfor annotation file, plot seq Logo of actual motif")
    parser_logo.add_argument("output", type=str,  help="pdf/png name")
    parser_logo.add_argument("-t", "--type", type=str, default='motif', help="motif / annotation")
    parser_logo.add_argument("-f", "--format", type=str, default='pdf', help="pdf/png")
    parser_logo.set_defaults(func=run_logo)

    # ------------------------------------------------------------
    # identity
    # ------------------------------------------------------------
    parser_identity = subparsers.add_parser('identity', 
                                            description='VAMPIRE identity v0.2.0\n'
                                                'Usage: vampire identity [options] [input prefix] [output_prefix]\n'
                                                'For example: vampire identity [input prefix] [output_prefix]\n',
                                            help='Calculate the identity of the tandem repeats.')
    parser_identity.add_argument("prefix", type=str, help="prefix of the input file")
    parser_identity.add_argument("output", type=str, help="output prefix")
    parser_identity.add_argument("-w", "--window-size", type=int, default=100, help="window size (unit: motif)")
    parser_identity.add_argument("-t", "--thread", type=int, default=30, help="thread number")
    parser_identity.add_argument("--mode", type=str, default='raw', help="mode: raw or invert")
    parser_identity.add_argument("--max-indel", type=int, default=0, help="maximum indel length")
    parser_identity.add_argument("--min-indel", type=int, default=0, help="minimum indel length")
    parser_identity.set_defaults(func=run_identity)

    # ------------------------------------------------------------
    # plotheatmap
    # ------------------------------------------------------------
    '''parser_plotheatmap = subparsers.add_parser('plotheatmap', help='Plot the heatmap of the tandem repeats.')
    parser_plotheatmap.add_argument('--input', required=True, help='Input directory')
    parser_plotheatmap.add_argument('--output', required=True, help='Output file')
    parser_plotheatmap.set_defaults(func=run_plotheatmap)'''


    
    args = parser.parse_args()
    args.func(args, parser)

if __name__ == '__main__':
    main()







'''def get_scripts():
    return {
        os.path.splitext(f)[0]: os.path.join(SCRIPT_DIR, f)
        for f in os.listdir(SCRIPT_DIR)
        if f.endswith(".py") and not f.startswith(".")
    }

def build_parser(script_map):
    parser = argparse.ArgumentParser(
        prog="vampire",
        description="🧛 VAMPIRE Toolkit - Unified interface for all internal scripts"
    )
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available tools")

    for name, path in script_map.items():
        sub = subparsers.add_parser(name, help=f"Run {name}.py script")
        sub.add_argument("args", nargs=argparse.REMAINDER, help="Arguments passed to the script")
        sub.set_defaults(script_path=path)

    return parser'''

'''def main():
    script_map = get_scripts()
    parser = build_parser(script_map)
    args = parser.parse_args()

    # 构建执行命令
    cmd = [sys.executable, args.script_path] + args.args
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Script {args.command} failed with return code {e.returncode}")
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()'''