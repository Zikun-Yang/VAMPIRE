import sourmash
import re
import numpy as np
import warnings
from Bio import SeqIO
from itertools import compress
from collections import Counter
from sklearn.cluster import DBSCAN, OPTICS
from itertools import combinations


def rolling_same(seq1: str, seq2: str) -> bool:
    """Check if seq2 is a rotation of seq1."""
    if len(seq1) != len(seq2):
        return False
    seq1double = seq1 * 2
    return seq2 in seq1double

class Estimate:
    def __init__(self, fasta_path: str, sample_rate: float, ksize_list: list, sampled_window_length: int, 
                 min_length = 10e3, max_length = 50e3):
        self.fasta_path = fasta_path
        self.sample_rate = sample_rate
        self.min_length = min_length
        self.max_length = max_length
        self.win_length = sampled_window_length
        self.sampled_sequence = self.sample()
        self.likely_length = self.get_likely_repeat_length(self.sampled_sequence, ksize_list)
        self.is_biased, self.base_proportions = self.calculate_base_composition(self.sampled_sequence)
        print(f'Detected possible repeat length: {self.likely_length} bp')
        print(f'Base composition: {self.base_proportions}')
        print(f'Is biased: {self.is_biased}')
        self.proper_k = self.get_likely_repeat_motif(self.sampled_sequence, self.likely_length)
        print(f'selected k: {self.proper_k} bp')
        

    def sample(self):
        """Sample sequences from the FASTA file."""
        min_sample_num = max(int(self.min_length / self.win_length) + 1, 1)
        max_sample_num = int(self.max_length / self.win_length)

        with open(self.fasta_path, "r") as handle:
            records = list(SeqIO.parse(handle, "fasta"))

        for record in records:
            seq = str(record.seq).upper()
            # Filter N character
            mask = [base != "N" for base in seq]
            seq = ''.join(compress(seq, mask))
            seq_length = len(seq)
            # Decide sampled window number
            sample_num = int(seq_length * self.sample_rate / self.win_length) + 1
            sample_num = max(min(sample_num, max_sample_num), min_sample_num)
            # Sample
            sampled_sequence = []
            if seq_length - self.win_length + 1 > 0:
                start = np.random.randint(0, seq_length - self.win_length + 1, size = sample_num)
                sampled_sequence.extend( [seq[s : s + self.win_length] for s in start] )
            else:
                sampled_sequence.append(seq)

        return sampled_sequence

    def get_likely_repeat_length(self, sequence, ksize_list):
        """Estimate the likely repeat length from the sampled sequences."""
        result = {}

        for k in ksize_list:
            for seq in sequence:
                mh = sourmash.MinHash(n = 5000, ksize = k)
                kmers = [kmer for kmer, _ in mh.kmers_and_hashes(seq, force=True) ]
                kmer_count = Counter(kmers)
                # filter
                max_count = kmer_count.most_common(1)[0][1]
                min_count = int(max_count * 0.05)
                kmer_count = [ [k, v] for k, v in kmer_count.items() if v >= min_count ]
                # get key kmer
                kmer_count.sort(key=lambda item: item[1], reverse=True)
                kmer_count = kmer_count[0:5]
                kmers = [kmer[0] for kmer in kmer_count]
                # get index
                for kmer in kmers:
                    matches = re.finditer(kmer, seq)
                    index = np.array([match.start() for match in matches])
                    first_diff = np.diff(index, n=1)

            if len(first_diff) > 3:
                # clustering
                data = np.array(first_diff).reshape(-1, 1)
                # set optics
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=RuntimeWarning)
                    optics = OPTICS(min_samples=3)
                    labels = optics.fit_predict(data)

                # Calculate the size of each cluster
                unique_labels, counts = np.unique(labels, return_counts=True)
                total_points = len(first_diff)

                # Filter out clusters with less than 50% of the total points
                for label, count in zip(unique_labels, counts):
                    ### print(label, count)
                    if count >= 0.2 * total_points:
                        # Find mode
                        values, counts = np.unique(first_diff[labels == label], return_counts=True)
                        mode_value = int(values[np.argmax(counts)])
                        if mode_value not in result.keys():
                            result[mode_value] = int(np.max(counts))
                        else:
                            result[mode_value] += int(np.max(counts))

                # Visualize the remaining clusters
                '''jitter = np.random.normal(0, 0.05, len(first_diff))  # Add small random noise for jitter
                y_values = [0] * len(first_diff) + jitter  # Apply jitter to the y-coordinates
                plt.scatter(first_diff, y_values, c=labels, cmap='viridis', marker='o', edgecolor='k')
                plt.title(f'Scatter Plot of First Differences with Jitter, k = {k}')
                plt.xlabel('First Differences')
                plt.ylabel('Y-axis (Jittered)')
                plt.colorbar(label='Labels')
                plt.grid(True, linestyle='--', alpha=0.7)
                plt.savefig(f'k{k}.pdf', bbox_inches='tight')
                plt.close()'''

                
        if result:
            # Sort by frequency and value
            sorted_values = sorted(result.keys(), key=lambda x: (-result[x], x))
            # Check for multiples
            final_values = []
            to_del = []
            for value in sorted_values:
                is_multiple = any(abs(float(value) / v - round(value / v)) <= 0.1 and 
                                  v != value for v in final_values)
                if is_multiple:
                    to_del.append(value)
                else:
                    final_values.append(value)
                
            for value in to_del:
                del result[value]

            max_combo_size = 3
            to_del = set()
            for r in range(2, max_combo_size + 1):
                for combo in combinations(result.keys(), r):
                    combo_sum = sum(combo)
                    if combo_sum in result.keys():
                        to_del.update(combo)
            
            for value in list(to_del):
                del result[value]
            
            sorted_values = sorted(result.keys(), key=lambda x: (-result[x], x))
            if result:
                return sorted_values[0]  # Return top

        return None
    
    def calculate_base_composition(self, sequence):
        sequence = ''.join(sequence)
        total_length = len(sequence)
        base_counts = {
            "A": sequence.count("A"),
            "T": sequence.count("T"),
            "C": sequence.count("C"),
            "G": sequence.count("G"),
        }
        base_proportions = {base: count / total_length for base, count in base_counts.items()}
        
        # 1 base > 50%
        has_bias = any(prop > 0.5 for prop in base_proportions.values())
        # 2 base > 80%
        if has_bias:
            return base_proportions, has_bias
        else:
            values_list = list(base_proportions.values())
            values_list.sort(reverse=True)
            if values_list[0] + values_list[1] > 0.8:
                has_bias = True
            else:
                has_bias = False
        return has_bias, base_proportions

    def get_likely_repeat_motif(self, sequence, length):
        kmer_dict = {}
        for seq in sequence:
            mh = sourmash.MinHash(n = 5000, ksize = length)
            kmers = [kmer for kmer, _ in mh.kmers_and_hashes(seq, force=True) ]
            kmer_count = Counter(kmers)
            # filter
            max_count = kmer_count.most_common(1)[0][1]
            min_count = int(max_count * 0.05)
            kmer_count = [ [k, v] for k, v in kmer_count.items() if v >= min_count ]
            # get key kmer
            kmer_count.sort(key=lambda item: item[1], reverse=True)
            kmer_count = kmer_count[0:5]
            for item in kmer_count:
                if item[0] not in kmer_dict.keys():
                    kmer_dict[item[0]] = item[1]
                else:
                    kmer_dict[item[0]] += item[1]

        # sort
        kmer_dict = dict(sorted(kmer_dict.items(), key=lambda item: item[1], reverse=True))
        ###print(kmer_dict)
        ###print(list(kmer_dict.keys()))
        # remove duplicated
        if len(kmer_dict.keys()) > 1:
            candidate_list = list(kmer_dict.keys())
            for i in range(1, len(candidate_list)):
                for j in range(i):
                    if rolling_same(candidate_list[i], candidate_list[j]):
                        kmer_dict[candidate_list[j]] += kmer_dict[candidate_list[i]]
                        del kmer_dict[candidate_list[i]]
                        break
        ###print(kmer_dict)

        motif_list = kmer_dict.keys()
        mink, maxk = 5, min(length, 31)   # 5 ~ 31

        if self.is_biased:
            if self.likely_length > 50:
                mink = 17
            elif self.likely_length > 15:
                mink = 13
            else:
                mink = 9

        if maxk <= mink:
            return mink

        # test when k = mink
        is_ok = True
        for motif in motif_list:
            seq = motif + motif[:mink - 1]
            mh = sourmash.MinHash(n = 5000, ksize = mink)
            kmers = [kmer for kmer, _ in mh.kmers_and_hashes(seq, force=True) ]
            kmer_count = Counter(kmers)
            max_count = kmer_count.most_common(1)[0][1]
            if max_count > 1:
                is_ok = False
                break
        if is_ok:
            return mink

        # find proper k
        while True:
            print(f'mink {mink}, maxk {maxk}')
            k = (mink + maxk) // 2
            is_ok = True
            for motif in motif_list:
                seq = motif + motif[:k - 1]
                mh = sourmash.MinHash(n = 5000, ksize = k)
                kmers = [kmer for kmer, _ in mh.kmers_and_hashes(seq, force=True) ]
                kmer_count = Counter(kmers)
                max_count = kmer_count.most_common(1)[0][1]
                if max_count > 1:
                    is_ok = False
                    break
            if is_ok:
                print(f'maxk: {maxk} -> {k}')
                maxk = k
            else:
                print(f'mink: {mink} -> {k}')
                mink = k

            if mink + 1 == maxk:
                k = maxk
                break

        return k
        
    def get_k(self):
        return self.proper_k




if __name__ == "__main__":
    # Example
    fasta_path = 'example/bo_chr2a_14110848-14463880_1-30000.fasta'
    #fasta_path = 'example/5bp_TR.fasta'
    sample_rate = 0.01
    sampled_window_length = 5000
    ksize_list = [3, 5, 9, 11, 15, 31, 41]
    min_length, max_length = 10e3, 50e3

    suggested_par = Estimate(fasta_path, sample_rate, ksize_list, sampled_window_length)
