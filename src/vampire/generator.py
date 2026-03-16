from typing import Any

def run_generator(cfg: dict[str, Any]) -> None:
    '''
    description: generate tandem repeat sequences
    Isnput:
        motifs: list[str]
        length: int
        mutation_rate: float
        seed: int
        prefix: str
    '''

    from vampire.generator_utils import TR_singleMotif, TR_multiMotif

    motif_list_len = len(cfg['motifs'])
    motifs = cfg['motifs']

    # check invalid characters
    for motif in motifs:
        if not all(c in 'ACGT' for c in motif):
            raise ValueError("ERROR: Invalid characters in motif!")

    if motif_list_len == 1:
        tr = TR_singleMotif(motifs[0], cfg['length'], cfg['mutation_rate'], cfg['seed'])
    else:
        tr = TR_multiMotif(motifs, cfg['length'], cfg['mutation_rate'], cfg['seed'])

    tr.save_seq_and_anno(cfg['prefix'])
