###########################################################

import random
from Bio import pairwise2
from Bio.pairwise2 import format_alignment
import time
import csv
import uuid
import tracemalloc
from Bio.Align.Applications import ClustalwCommandline         ###################  All Imports
from Bio.Align.Applications import MuscleCommandline
import pandas as pd
from scipy import stats
import subprocess
import os 
import matplotlib.pyplot as plt
from scipy.stats import gumbel_r
import numpy as np
from Bio.Align import substitution_matrices
##########################################################
class NeedleMan:

    def __init__(self, length, rate, match_rew, mismatch_pen, gap_pen):
        self.length = length
        self.string = None
        self.rate = rate
        self.original_sequence = None
        self.mutated_sequence = None
        self.mutations = None
        self.movements = None
        self.matrix_r = None
        self.matrix_t = None
        self.alignment = None
        self.score = None
        self.ss = None
        self.biopython_score = None
        self.biopython_time = None
        self.biopython_memory = None
        self.biopython_ss = None
        self.clustal_score = None
        self.clustal_time = None                                        ####################### Setting up class variables unique to each run 
        self.clustal_memory = None
        self.clustal_ss = None
        self.muscle_score = None
        self.muscle_time = None
        self.muscle_memory = None
        self.muscle_ss = None
        self.smith_score = None
        self.smith_time = None
        self.smith_memory = None
        self.smith_ss = None
        self.gap_pen = gap_pen
        self.match_rew = match_rew
        self.mismatch_pen = mismatch_pen
        self.tt_matrix = {
            ('a', 'a'): 2, ('g', 'g'): 2, ('c', 'c'): 2, ('t', 't'): 2,
            ('a', 'g'): 1, ('g', 'a'): 1,  
            ('c', 't'): 1, ('t', 'c'): 1,                                     ### transition/transversion matrix
            ('a', 'c'): -1, ('a', 't'): -1, ('g', 'c'): -1, ('g', 't'): -1,
            ('c', 'a'): -1, ('t', 'a'): -1, ('c', 'g'): -1, ('t', 'g'): -1,
        }
        self.nuc44 = substitution_matrices.load("NUC.4.4")                            ####NUC4.4 matrix

##################################################################################################


    def create_sequence(self):
        characters = ['a', 'c', 'g', 't']
        self.original_sequence = ''.join(random.choices(characters, k=self.length))  #### random sequence of length set at the start of the class

    def second_rando(self):
        characters = ['a', 'c', 'g', 't']
        self.mutated_sequence = ''.join(random.choices(characters, k=self.length)) #### second random sequence only used to make the null distributions

    def add_errors(self):
        rate = int(len(self.original_sequence) * (self.rate / 100)) #percentage of sequence length to be mutated
        substitution_rate = rate // 2  #get this as a whole number 
        string = list(self.original_sequence)
        characters = ['a', 'c', 'g', 't']
        self.mutations = {} # set charactrs and start dict 
        total_indexes = random.sample(range(len(string)), rate) # random places in the sequence
        substitution_indexes = total_indexes[:substitution_rate]  #split hald between subs
        indel_indexes = total_indexes[substitution_rate:] # and half for indels
        for index in substitution_indexes:
            current_char = string[index]
            new_char = random.choice([char for char in characters if char != current_char]) # for each randomised location get a new character for subs
            self.mutations[index] = f"Substitution: {string[index]} -> {new_char}"
            string[index] = new_char
        for index in sorted(indel_indexes, reverse=True):  
            if random.choice([True, False]):  # random 50/50 between insertions and deletions
                new_char = random.choice(characters)
                self.mutations[index] = f"Insertion: {new_char}"
                string.insert(index, new_char)
            else:
                deleted_char = string[index]
                self.mutations[index] = f"Deletion: {deleted_char}"
                del string[index]
        self.mutated_sequence = ''.join(string) # returnthe mutated string

    def create_matrix(self):
        original_sequence = list(self.original_sequence) # x-axis of needleman wunsch grid being created 
        matrix_r = []
        matrix_r.append("*")
        matrix_r.append("*")
        for letter in original_sequence:
            matrix_r.append(letter)
        self.matrix_r = matrix_r
        
        mutated_sequence = list(self.mutated_sequence)    # y-axis of needleman-wunsch grid being created
        matrix_t = []
        placehold = []
        matrix_t.append(placehold)
        for letter in mutated_sequence:
            placehold = []
            matrix_t.append(placehold)
        matrix_t[0].append("*")
        matrix_t[0].append("0")
        for i in range(1, len(matrix_r) - 1):
            matrix_t[0].append(f"-{i}")
        for i, j in enumerate(mutated_sequence):
            placehold = []
            placehold.append(j)
            matrix_t[i+1] = placehold
        for item in matrix_t:
            if not item[0] == "*":
                equivindex = matrix_t.index(item) 
                item.append(f"-{equivindex}")
        

        
        for item in matrix_t:
            if not item[0] == "*":
                length_diff = len(matrix_r) - len(item) # fill missing 'cells' i.e. place in list of lists with blank placeholder
                for i in range(length_diff):
                    item.append(" ")

        
        self.movements = []
        for item in matrix_t:
            if not item[0] == "*":
                item_place = matrix_t.index(item)
                matrix_t[item_place], moves = linewise_matrix_fill(item, matrix_r, matrix_t[item_place -1], item_place) # call 'linewise matrix fill' i.e. max() equation from originalpaper for each cell 
                self.movements.extend(moves)
 
        self.matrix_t = matrix_t # finally, assign to self so other methods can use it 


    def print_matrix(self):
        for item in self.matrix_t:
            for bit in item[1:]:
                if len(str(bit)) == 1:                    # print the matrix row by row - note this was only used for validation and had no real purpose
                    item[item.index(bit)] = f" {bit}"
                else:
                    item[item.index(bit)] = f"{bit}"


    def assemble_sequences(self):
        matrix = []
        for item in self.matrix_t:
            for bit in item:
                try:
                    item[item.index(bit)] = int(bit)
                except:
                    pass                                           # set up and call traceback logic function 
            self.matrix_t[self.matrix_t.index(item)] = item
        for item in self.matrix_t: 
            matrix.append(item[1:])
        self.alignment = get_optimal_alignment(matrix, self.mutated_sequence, self.original_sequence) # call it and assign alignment tuple to self so other methods can access 

    def compare_alignment(self):
        aligned_mutant = self.alignment[0]
        aligned_original = self.alignment[1]
        aligned_mutant = list(aligned_mutant)
        aligned_original = list(aligned_original) # turn alignments into lists so side by side comparison can be made 
        score = 0
        for i, item in enumerate(aligned_mutant):  # use the index directly
            if i < len(aligned_original) and aligned_original[i] == item:  # ensure valid index
                score += self.match_rew
            elif i < len(aligned_original) and aligned_original[i] == "-" or item == "-": # manual code to score each alignment for comparing mine and biopythons to make sure doing it the same way 
                score -= self.gap_pen 
            else:
                score -= self.mismatch_pen
        self.score = score


    def biopython(self):
        start_time = time.time()
        tracemalloc.start() # time it
        #alignments = pairwise2.align.globalxx(self.original_sequence, self.mutated_sequence) # core calling 
        #alignments = pairwise2.align.globalms(self.original_sequence, self.mutated_sequence, open=-10, extend=-0.5, match=1,mismatch=-1)  #calling for gap penalty comparisons 
        alignments = pairwise2.align.globalds(self.original_sequence.upper(), self.mutated_sequence.upper(), self.nuc44, open=0, extend=0) # calling for scoring matrix comparisons 
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        time_diff = end_time - start_time
        self.biopython_time = time_diff
        self.biopython_memory = peak
        alignment = format_alignment(*alignments[0])
        alignment = alignment.split("\n")
        top_alignment = alignments[0]
        aligned_mutant = list(alignment[2])
        aligned_original = list(alignment[0])
        score = 0
        for i, item in enumerate(aligned_mutant):  
            if i < len(aligned_original) and aligned_original[i] == item:  
                score += self.match_rew
            elif i < len(aligned_original) and aligned_original[i] == "-" or item == "-": #manual scoring if needed 
                score -= self.gap_pen
            else:
                score -= self.mismatch_pen
        self.biopython_score = score
        self.biopython_ss = self.specificity_and_sensitivity_with_tolerance(aligned_mutant, aligned_original, "bio") # get specificity and sensitivity

    def clustal(self):
        input_text = f""">Original\n{self.original_sequence}\n>Mutated\n{self.mutated_sequence}"""
        with open("clustal_input.fasta", "w") as input:
            input.write(input_text)
        input_file = "clustal_input.fasta"
        output_file = "clustal_output.fasta" # file import 
        start_time = time.time()
        tracemalloc.start()
        clustalomega_cline = ClustalwCommandline("clustalw", infile=input_file, outfile=output_file, gapopen=-10, gapext=-0.5) # gapping calling 
        #clustalomega_cline = ClustalwCommandline("clustalw", infile=input_file, outfile=output_file) # normal / default calling 
        stdout, stderr = clustalomega_cline()
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        time_diff = end_time - start_time
        self.clustal_time = time_diff
        self.clustal_memory = peak
        aligned_original = ""
        aligned_mutant = ""
        with open("clustal_output.fasta", "r") as output: # for clustal you have to specify an output file 
            lines = output.readlines()
        for line in lines:
            if line.startswith("Original"):
                aligned_original += line[16:].strip() # reading and processing the output file 
            elif line.startswith("Mutated"):
                aligned_mutant += line[16:].strip()
        score = 0
        for i, item in enumerate(aligned_mutant):  
            if i < len(aligned_original) and aligned_original[i] == item:  
                score += self.match_rew
            elif i < len(aligned_original) and aligned_original[i] == "-" or item == "-":
                score -= self.gap_pen
            else:
                score -= self.mismatch_pen
        self.clustal_score = score
        self.clustal_ss = self.specificity_and_sensitivity_with_tolerance(aligned_mutant, aligned_original, "clustal") # get spec and sen 

    def muscle(self):
        input_text = f""">Original\n{self.original_sequence}\n>Mutated\n{self.mutated_sequence}"""
        with open("muscle_input.fasta", "w") as input:
            input.write(input_text)
        input_file = "muscle_input.fasta"
        output_file = "muscle_output.fasta"
        start_time = time.time()
        tracemalloc.start()

        muscle_path = os.path.expanduser("/usr/local/bin/muscle3.8.31_i86linux64") # where muscle is downloaded
        #muscle_command = [muscle_path, "-in", input_file, "-out", output_file] # default use 
        muscle_command = [muscle_path, "-in", input_file, "-out", output_file, "-gapopen", "-10", "-gapextend", "-0.5"] # gapping use 
        subprocess.run(muscle_command, check=True, capture_output=True, text=True) # running the command 

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        time_diff = end_time - start_time
        self.muscle_time = time_diff
        self.muscle_memory = peak

        with open("muscle_output.fasta", "r") as output:
            lines = output.read()
        lines = lines.split("\n")
        seq2 = lines.index(">Mutated")
        aligned_original = "".join(lines[1:seq2])
        aligned_mutant = "".join(lines[seq2 + 1:]) # processing output file 


        score = 0
        for i, item in enumerate(aligned_mutant):  
            if i < len(aligned_original) and aligned_original[i] == item:  
                score += self.match_rew
            elif i < len(aligned_original) and aligned_original[i] == "-" or item == "-": #manual scoring
                score -= self.gap_pen
            else:
                score -= self.mismatch_pen

        self.muscle_score = score
        self.muscle_ss = self.specificity_and_sensitivity_with_tolerance(aligned_mutant, aligned_original, "muscle") # getting spec and sen

    def smith(self):
        start_time = time.time()
        tracemalloc.start()
        #alignments = pairwise2.align.localxx(self.original_sequence, self.mutated_sequence) # default run 
        #alignments = pairwise2.align.localms(self.original_sequence, self.mutated_sequence, open=-10, extend=-0.5, match=1, mismatch=-1) # gapping run 
        alignments = pairwise2.align.localds(self.original_sequence.upper(), self.mutated_sequence.upper(), self.nuc44, 0, 0) # scoring run 
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        end_time = time.time()
        time_diff = end_time - start_time
        self.smith_time = time_diff
        self.smith_memory = peak

        top_alignment = alignments[0]
        score = top_alignment.score
        alignment = format_alignment(*alignments[0])
        alignment = alignment.split("\n")

        aligned_mutant = list(alignment[2])
        aligned_original = list(alignment[0])
        self.smith_score = score
        self.smith_ss = self.specificity_and_sensitivity_with_tolerance(aligned_mutant, aligned_original, "smith")



    def specificity_and_sensitivity(self, aligned_mutated, aligned_original, aligner):
        mutations_dict = self.mutations
        TP = FP = FN = TN = 0 # initiate the values
        aligned_original = list(aligned_original.lower())
        aligned_mutated = list(aligned_mutated.lower()) # turn everything lower case in case it wasnt before
        mutation_positions = set(mutations_dict.keys()) # get the positions from the index: location dictionary 
        for i in range(len(aligned_original)): # iterate throught the sequence
            orig = aligned_original[i]
            mut = aligned_mutated[i]
            if i in mutation_positions:
                mutation = mutations_dict[i]
                if "Substitution" in mutation: # resolve substitutions
                    _, change = mutation.split(": ")
                    from_base, to_base = change.split(" -> ")
                    if orig == from_base and mut == to_base: # do they match?
                        TP += 1
                    else:
                        FN += 1
                elif "Insertion" in mutation:
                    _, inserted_base = mutation.split(": ") # ""
                    if orig == '-' and mut == inserted_base:
                        TP += 1
                    else:
                        FN += 1
                elif "Deletion" in mutation:
                    _, deleted_base = mutation.split(": ") # ""
                    if orig == deleted_base and mut == '-':
                        TP += 1
                    else:
                        FN += 1
            else:
                if orig == mut: # final logic for TN and FP 
                    TN += 1
                else:
                    FP += 1
        try:
            sensitivity = TP / (TP + FN)
        except ZeroDivisionError: # try to calculate values
            sensitivity = "NA"
        try:
            specificity = TN / (TN + FP)
        except ZeroDivisionError:
            specificity = "NA"

        return specificity, sensitivity


def get_optimal_alignment(matrix, sequence1, sequence2):
    sideways = len(sequence1)
    upwards = len(sequence2)
    aligned_seq_1 = " "
    aligned_seq_2 = " "

    while sideways >0 or upwards > 0:
        if sequence1[sideways-1] == sequence2[upwards-1]:
            diagonal_cell = matrix[sideways-1][upwards-1] + 1
        else: 
            diagonal_cell = matrix[sideways-1][upwards-1] - 1

        side_cell = matrix[sideways-1][upwards] - 1

        if sideways > 0 and upwards > 0 and matrix[sideways][upwards] == diagonal_cell: # looking at each cell and deducing from which direction it came from; reassembling the sequences from that
            aligned_seq_1 = sequence1[sideways-1] + aligned_seq_1
            aligned_seq_2 = sequence2[upwards-1] + aligned_seq_2
            sideways -=1
            upwards -= 1
        
        elif sideways > 0 and matrix[sideways][upwards] == side_cell:
            aligned_seq_1 = sequence1[sideways-1] + aligned_seq_1
            aligned_seq_2 = '-' + aligned_seq_2
            sideways -= 1

        else:
            aligned_seq_1 = '-' + aligned_seq_1
            aligned_seq_2 = sequence2[upwards-1] + aligned_seq_2
            upwards -= 1
    
    return aligned_seq_1, aligned_seq_2




def match_or_mismatch(char1, char2):
    return 1 if char1 == char2 else -1

def gap_penalty():
    return -1

def linewise_matrix_fill(line, matrix_r, prior, y):
    linebase = line[0]
    movements = []
    for item in line: #setting up variables
        if item == " ":
            place = line.index(item)
            itemcoordx = place
            itemcoordy = int(y) +1
            itemcoord = f"{itemcoordx},{itemcoordy}"
            refbase = matrix_r[place]
            if linebase == refbase:
                modifier = +1
            else:
                modifier = -1
            left = int(line[place -1])
            leftcoord = f"{int(itemcoordx) -1},{itemcoordy}"
            up = int(prior[place])
            upcoord = f"{itemcoordx},{int(itemcoordy) -1}"
            diagonal = int(prior[place -1])
            diagonalcoord = f"{int(itemcoordx) -1},{int(itemcoordy) -1}"

            #filling line
            leftnum = left -1
            upnum = up -1
            diagonalnum = diagonal + modifier
            three = [leftnum, upnum, diagonalnum]
            fill = max(three)
            line[place] = fill
            if fill == leftnum:
                movements.append(f"{leftcoord}->{itemcoord}")
            if fill == upnum:
                movements.append(f"{upcoord}->{itemcoord}")
            if fill == diagonalnum:
                movements.append(f"{diagonalcoord}->{itemcoord}")
    
    return line, movements


###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
######################################################################## FROM HERE IS CALLING THE ALIGNERS TO GET DATA ###################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################
###########################################################################################################################################################################################

#BIOPYTHON NEEDLEMAN-WUNSCH VS CUSTOM NEEDLEMAN-WUNSCH 

def my_and_biopythons_null_distribution_order():
    nullrun1 = NeedleMan(100, 10, 1, 1, 1)
    nullrun1.create_sequence()
    nullrun1.second_rando()
    start_time = time.time()
    tracemalloc.start()
    nullrun1.create_matrix()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    #short_seq.print_matrix()
    nullrun1.assemble_sequences()
    nullrun1.compare_alignment()
    time_diff = end_time - start_time
    nullrun1.biopython()
    return nullrun1.score, nullrun1.biopython_score

def my_and_biopythons_null_distribution_execute():
    with open('my_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('biopython_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = my_and_biopythons_null_distribution_order()
        with open('my_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('biopython_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        print(i)

def my_and_biopython_compare_order(mutation_rate):
    compare1 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare1.create_sequence()
    compare1.add_errors()
    start_time = time.time()
    tracemalloc.start()
    compare1.create_matrix()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    compare1.assemble_sequences()
    compare1.compare_alignment()
    time_diff = end_time - start_time
    compare1.ss = compare1.specificity_and_sensitivity(compare1.alignment[0], compare1.alignment[1])
    compare1.biopython()
    return compare1.score, time_diff, peak, compare1.ss, compare1.biopython_score, compare1.biopython_time, compare1.biopython_memory, compare1.biopython_ss

def my_and_biopython_compare_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Score", "Time", "Peak Memory", "SS",
        "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS"
    ]
    
    with open('my_and_biopythons_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = my_and_biopython_compare_order(mutation_rate)
            current_dict.update({
                "Score": results[0],
                "Time": results[1],
                "Peak Memory": results[2],
                "SS": results[3],
                "Biopython Score": results[4],
                "Biopython Time": results[5],
                "Biopython Memory": results[6],
                "Biopython_SS": results[7]
            })
            
            with open('my_and_biopythons_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)

def add_z_values_to_my_bio_compare():
    my_algo_null = "my_aligner_null_distribution.csv"
    bio_algo_null = "biopython_aligner_null_distribution.csv"
    comparison_data = "my_and_biopythons_compare.csv"

    mynulldf = pd.read_csv(my_algo_null)
    mynulldata = mynulldf['Score']
    mynulldatamean = mynulldata.mean()
    mynulldatastdev = mynulldata.std()
    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Score', 'Biopython Score']].mean()
    mean_scores['My Z-Score'] = (mean_scores['Score'] - mynulldatamean) / mynulldatastdev
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev

    output_file = "my_and_biopythons_compare_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores

def my_and_biopython_compare_scale_order(scale):
    compare1 = NeedleMan(scale, 10, 1, 1, 1)
    compare1.create_sequence()
    compare1.add_errors()
    start_time = time.time()
    tracemalloc.start()
    compare1.create_matrix()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    compare1.assemble_sequences()
    compare1.compare_alignment()
    time_diff = end_time - start_time
    compare1.ss = compare1.specificity_and_sensitivity(compare1.alignment[0], compare1.alignment[1])
    compare1.biopython()
    return compare1.score, time_diff, peak, compare1.ss, compare1.biopython_score, compare1.biopython_time, compare1.biopython_memory, compare1.biopython_ss

def my_and_biopython_compare_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Score", "Time", "Peak Memory", "SS",
        "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS"
    ]
    
    with open('my_and_biopython_compare_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = my_and_biopython_compare_scale_order(scale)
            current_dict.update({
                "Score": results[0],
                "Time": results[1],
                "Peak Memory": results[2],
                "SS": results[3],
                "Biopython Score": results[4],
                "Biopython Time": results[5],
                "Biopython Memory": results[6],
                "Biopython_SS": results[7]
            })
            
            with open('my_and_biopython_compare_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)

def solo_5000(scale):
    compare1 = NeedleMan(scale, 10, 1, 1, 1)
    compare1.create_sequence()
    compare1.add_errors()
    start_time = time.time()
    tracemalloc.start()
    compare1.create_matrix()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    end_time = time.time()
    compare1.assemble_sequences()
    compare1.compare_alignment()
    time_diff = end_time - start_time
    compare1.ss = compare1.specificity_and_sensitivity(compare1.alignment[0], compare1.alignment[1])
    return compare1.score, time_diff, peak, compare1.ss


 
#################################################
#################################################
###############COMPARING ALL ALIGNERS ##################################
#################################################
#################################################
#################################################

def create_null_distribution_order():
    nullrun2 = NeedleMan(100, 10, 1, 1, 1)
    nullrun2.create_sequence()
    nullrun2.second_rando()
    
    nullrun2.biopython()
    nullrun2.clustal()
    nullrun2.muscle()
    nullrun2.smith()
    return nullrun2.biopython_score, nullrun2.clustal_score, nullrun2.muscle_score, nullrun2.smith_score



def create_null_distribution_execute():

    with open('TEST2biopython_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2clustal_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2muscle_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2smith_aligner_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = create_null_distribution_order()

        with open('TEST2biopython_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('TEST2clustal_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        with open('TEST2muscle_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[2]])
        with open('TEST2smith_aligner_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[3]])
        print(i)


def all_aligners_first_compare_mutation_order(mutation_rate):
    compare2 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss



def all_aligners_first_compare_mutation_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = all_aligners_first_compare_mutation_order(mutation_rate)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)


def all_aligners_first_compare_scale_order(scale):
    compare2 = NeedleMan(scale, 10, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss


def all_aligners_first_compare_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_compare_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = all_aligners_first_compare_scale_order(scale)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_compare_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)



def add_z_values_to_first_comparison():
    bio_algo_null = "TEST2biopython_aligner_null_distribution.csv"
    clustal_algo_null = "TEST2clustal_aligner_null_distribution.csv"
    muscle_algo_null = "TEST2muscle_aligner_null_distribution.csv"
    smith_algo_null = "TEST2smith_aligner_null_distribution.csv"
    comparison_data = "all_aligners_first_compare.csv"

    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    clustalnulldf = pd.read_csv(clustal_algo_null)
    clustalnulldata = clustalnulldf['Score']
    clustalnulldatamean = clustalnulldata.mean()
    clustalnulldatastdev = clustalnulldata.std()

    musclenulldf = pd.read_csv(muscle_algo_null)
    musclenulldata = musclenulldf['Score']
    musclenulldatamean = musclenulldata.mean()
    musclenulldatastdev = musclenulldata.std()

    smithnulldf = pd.read_csv(smith_algo_null)
    smithnulldata = smithnulldf['Score']
    smithnulldatamean = smithnulldata.mean()
    smithnulldatastdev = smithnulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Biopython Score', 'Clustal Score', 'Muscle Score', 'Smith Score']].mean()
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev
    mean_scores['Clustal Z-Score'] = (mean_scores['Clustal Score'] - clustalnulldatamean) / clustalnulldatastdev
    mean_scores['Muscle Z-Score'] = (mean_scores['Muscle Score'] - musclenulldatamean) / musclenulldatastdev
    mean_scores['Smith Z-Score'] = (mean_scores['Smith Score'] - smithnulldatamean) / smithnulldatastdev

    output_file = "all_first_compare_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores




################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
###########################COMPARING ALL ALIGNERS WITH OPEN VS EXTEND GAP PENALTIES#############################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def create_null_distribution_OPEN_order():
    nullrun2 = NeedleMan(100, 10, 1, 1, 1)
    nullrun2.create_sequence()
    nullrun2.second_rando()
    
    nullrun2.biopython()
    nullrun2.clustal()
    nullrun2.muscle()
    nullrun2.smith()
    return nullrun2.biopython_score, nullrun2.clustal_score, nullrun2.muscle_score, nullrun2.smith_score

def create_null_distribution_OPEN_execute():

    with open('TEST2biopython_aligner_OPEN_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2clustal_aligner_OPEN_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2muscle_aligner_OPEN_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2smith_aligner_OPEN_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = create_null_distribution_OPEN_order()

        with open('TEST2biopython_aligner_OPEN_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('TEST2clustal_aligner_OPEN_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        with open('TEST2muscle_aligner_OPEN_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[2]])
        with open('TEST2smith_aligner_OPEN_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[3]])
        print(i)

def all_aligners_first_compare_mutation_OPEN_order(mutation_rate):
    compare2 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss

def all_aligners_first_compare_mutation_OPEN_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_OPEN_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = all_aligners_first_compare_mutation_OPEN_order(mutation_rate)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_OPEN_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)



def add_z_values_to_OPEN_comparison():
    bio_algo_null = "TEST2biopython_aligner_OPEN_null_distribution.csv"
    clustal_algo_null = "TEST2clustal_aligner_OPEN_null_distribution.csv"
    muscle_algo_null = "TEST2muscle_aligner_OPEN_null_distribution.csv"
    smith_algo_null = "TEST2smith_aligner_OPEN_null_distribution.csv"
    comparison_data = "all_aligners_first_OPEN_compare.csv"

    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    clustalnulldf = pd.read_csv(clustal_algo_null)
    clustalnulldata = clustalnulldf['Score']
    clustalnulldatamean = clustalnulldata.mean()
    clustalnulldatastdev = clustalnulldata.std()

    musclenulldf = pd.read_csv(muscle_algo_null)
    musclenulldata = musclenulldf['Score']
    musclenulldatamean = musclenulldata.mean()
    musclenulldatastdev = musclenulldata.std()

    smithnulldf = pd.read_csv(smith_algo_null)
    smithnulldata = smithnulldf['Score']
    smithnulldatamean = smithnulldata.mean()
    smithnulldatastdev = smithnulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Biopython Score', 'Clustal Score', 'Muscle Score', 'Smith Score']].mean()
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev
    mean_scores['Clustal Z-Score'] = (mean_scores['Clustal Score'] - clustalnulldatamean) / clustalnulldatastdev
    mean_scores['Muscle Z-Score'] = (mean_scores['Muscle Score'] - musclenulldatamean) / musclenulldatastdev
    mean_scores['Smith Z-Score'] = (mean_scores['Smith Score'] - smithnulldatamean) / smithnulldatastdev

    output_file = "all_compare_OPEN_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores



def create_null_distribution_EXTEND_order():
    nullrun2 = NeedleMan(100, 10, 1, 1, 1)
    nullrun2.create_sequence()
    nullrun2.second_rando()
    
    nullrun2.biopython()
    nullrun2.clustal()
    nullrun2.muscle()
    nullrun2.smith()
    return nullrun2.biopython_score, nullrun2.clustal_score, nullrun2.muscle_score, nullrun2.smith_score

def create_null_distribution_EXTEND_execute():

    with open('TEST2biopython_aligner_EXTEND_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2clustal_aligner_EXTEND_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2muscle_aligner_EXTEND_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST2smith_aligner_EXTEND_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = create_null_distribution_EXTEND_order()

        with open('TEST2biopython_aligner_EXTEND_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('TEST2clustal_aligner_EXTEND_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        with open('TEST2muscle_aligner_EXTEND_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[2]])
        with open('TEST2smith_aligner_EXTEND_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[3]])
        print(i)

def all_aligners_first_compare_mutation_EXTEND_order(mutation_rate):
    compare2 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss

def all_aligners_first_compare_mutation_EXTEND_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_EXTEND_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = all_aligners_first_compare_mutation_EXTEND_order(mutation_rate)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_EXTEND_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)



def add_z_values_to_EXTEND_comparison():
    bio_algo_null = "TEST2biopython_aligner_EXTEND_null_distribution.csv"
    clustal_algo_null = "TEST2clustal_aligner_EXTEND_null_distribution.csv"
    muscle_algo_null = "TEST2muscle_aligner_EXTEND_null_distribution.csv"
    smith_algo_null = "TEST2smith_aligner_EXTEND_null_distribution.csv"
    comparison_data = "all_aligners_first_EXTEND_compare.csv"

    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    clustalnulldf = pd.read_csv(clustal_algo_null)
    clustalnulldata = clustalnulldf['Score']
    clustalnulldatamean = clustalnulldata.mean()
    clustalnulldatastdev = clustalnulldata.std()

    musclenulldf = pd.read_csv(muscle_algo_null)
    musclenulldata = musclenulldf['Score']
    musclenulldatamean = musclenulldata.mean()
    musclenulldatastdev = musclenulldata.std()

    smithnulldf = pd.read_csv(smith_algo_null)
    smithnulldata = smithnulldf['Score']
    smithnulldatamean = smithnulldata.mean()
    smithnulldatastdev = smithnulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Biopython Score', 'Clustal Score', 'Muscle Score', 'Smith Score']].mean()
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev
    mean_scores['Clustal Z-Score'] = (mean_scores['Clustal Score'] - clustalnulldatamean) / clustalnulldatastdev
    mean_scores['Muscle Z-Score'] = (mean_scores['Muscle Score'] - musclenulldatamean) / musclenulldatastdev
    mean_scores['Smith Z-Score'] = (mean_scores['Smith Score'] - smithnulldatamean) / smithnulldatastdev

    output_file = "all_compare_EXTEND_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores



def all_aligners_first_compare_EXTEND_scale_order(scale):
    compare2 = NeedleMan(scale, 10, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss


def all_aligners_first_compare_EXTEND_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_compare_EXTEND_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = all_aligners_first_compare_EXTEND_scale_order(scale)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_compare_EXTEND_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)



def all_aligners_first_compare_OPEN_scale_order(scale):
    compare2 = NeedleMan(scale, 10, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    compare2.clustal()
    compare2.muscle()
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.clustal_score, compare2.clustal_time, compare2.clustal_memory, compare2.clustal_ss, compare2.muscle_score, compare2.muscle_time, compare2.muscle_memory, compare2.muscle_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss


def all_aligners_first_compare_OPEN_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Clustal Score", "Clustal Time", "Clustal Memory", "Clustal SS", "Muscle Score", "Muscle Time", "Muscle Memory", "Muscle SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_compare_OPEN_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = all_aligners_first_compare_OPEN_scale_order(scale)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Clustal Score": results[4], 
                "Clustal Time": results[5], 
                "Clustal Memory": results[6], 
                "Clustal SS": results[7], 
                "Muscle Score": results[8], 
                "Muscle Time": results[9], 
                "Muscle Memory": results[10], 
                "Muscle SS": results[11], 
                "Smith Score": results[12], 
                "Smith Time": results[13], 
                "Smith Memory": results[14], 
                "Smith SS": results[15]
            })
            
            with open('all_aligners_first_compare_OPEN_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)


################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
##############################COMPARE NEEDLEMAN-WUNSCH AND SMITH-WATERMAN WITH DIFFERENT MATRICES###############
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################

def create_null_distribution_TRANSVERSE_order():
    nullrun2 = NeedleMan(100, 10, 1, 1, 1)
    nullrun2.create_sequence()
    nullrun2.second_rando()
    
    nullrun2.biopython()
    nullrun2.smith()
    return nullrun2.biopython_score, nullrun2.smith_score

def create_null_distribution_TRANSVERSE_execute():

    with open('TEST3biopython_aligner_TRANSVERSE_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST3smith_aligner_TRANSVERSE_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = create_null_distribution_TRANSVERSE_order()

        with open('TEST3biopython_aligner_TRANSVERSE_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('TEST3smith_aligner_TRANSVERSE_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        print(i)

def all_aligners_first_compare_mutation_TRANSVERSE_order(mutation_rate):
    compare2 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss

def all_aligners_first_compare_mutation_TRANSVERSE_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_TRANSVERSE_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = all_aligners_first_compare_mutation_TRANSVERSE_order(mutation_rate)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Smith Score": results[4], 
                "Smith Time": results[5], 
                "Smith Memory": results[6], 
                "Smith SS": results[7]
            })
            
            with open('all_aligners_first_TRANSVERSE_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)

def add_z_values_to_TRANSVERSE_comparison():
    bio_algo_null = "TEST3biopython_aligner_TRANSVERSE_null_distribution.csv"
    smith_algo_null = "TEST3smith_aligner_TRANSVERSE_null_distribution.csv"
    comparison_data = "all_aligners_first_TRANSVERSE_compare.csv"

    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    smithnulldf = pd.read_csv(smith_algo_null)
    smithnulldata = smithnulldf['Score']
    smithnulldatamean = smithnulldata.mean()
    smithnulldatastdev = smithnulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Biopython Score','Smith Score']].mean()
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev
    mean_scores['Smith Z-Score'] = (mean_scores['Smith Score'] - smithnulldatamean) / smithnulldatastdev

    output_file = "all_compare_TRANSVERSE_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores



def all_aligners_first_compare_TRANSVERSE_scale_order(scale):
    compare2 = NeedleMan(scale, 10, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss


def all_aligners_first_compare_TRANSVERSE_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"]
    
    with open('all_aligners_first_compare_TRANSVERSE_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = all_aligners_first_compare_TRANSVERSE_scale_order(scale)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3],  
                "Smith Score": results[4], 
                "Smith Time": results[5], 
                "Smith Memory": results[6], 
                "Smith SS": results[7]
            })
            
            with open('all_aligners_first_compare_TRANSVERSE_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)

def create_null_distribution_NUC_order():
    nullrun2 = NeedleMan(100, 10, 1, 1, 1)
    nullrun2.create_sequence()
    nullrun2.second_rando()
    
    nullrun2.biopython()
    nullrun2.smith()
    return nullrun2.biopython_score, nullrun2.smith_score

def create_null_distribution_NUC_execute():

    with open('TEST3biopython_aligner_NUC_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])
    with open('TEST3smith_aligner_NUC_null_distribution.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Score"])

    for i in range(10000):
        run = create_null_distribution_NUC_order()

        with open('TEST3biopython_aligner_NUC_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[0]])
        with open('TEST3smith_aligner_NUC_null_distribution.csv', mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([run[1]])
        print(i)

def all_aligners_first_compare_mutation_NUC_order(mutation_rate):
    compare2 = NeedleMan(100, mutation_rate, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss

def all_aligners_first_compare_mutation_NUC_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"
    ]
    
    with open('all_aligners_first_NUC_compare.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for mutation_rate in range(1, 100):
        for i in range(100):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": 100,
                "Mutation Rate": mutation_rate
            }
            results = all_aligners_first_compare_mutation_NUC_order(mutation_rate)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3], 
                "Smith Score": results[4], 
                "Smith Time": results[5], 
                "Smith Memory": results[6], 
                "Smith SS": results[7]
            })
            
            with open('all_aligners_first_NUC_compare.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(i)

all_aligners_first_compare_mutation_NUC_execute()

def add_z_values_to_NUC_comparison():
    bio_algo_null = "TEST3biopython_aligner_NUC_null_distribution.csv"
    smith_algo_null = "TEST3smith_aligner_NUC_null_distribution.csv"
    comparison_data = "all_aligners_first_NUC_compare.csv"

    bionulldf = pd.read_csv(bio_algo_null)
    bionulldata = bionulldf['Score']
    bionulldatamean = bionulldata.mean()
    bionulldatastdev = bionulldata.std()

    smithnulldf = pd.read_csv(smith_algo_null)
    smithnulldata = smithnulldf['Score']
    smithnulldatamean = smithnulldata.mean()
    smithnulldatastdev = smithnulldata.std()

    comparedf = pd.read_csv(comparison_data)
    grouped = comparedf.groupby(comparedf.index // 100)
    mean_scores = grouped[['Biopython Score','Smith Score']].mean()
    mean_scores['Bio Z-Score'] = (mean_scores['Biopython Score'] - bionulldatamean) / bionulldatastdev
    mean_scores['Smith Z-Score'] = (mean_scores['Smith Score'] - smithnulldatamean) / smithnulldatastdev

    output_file = "all_compare_NUC_z_score.csv"
    mean_scores.to_csv(output_file, index=False)


    return mean_scores



def all_aligners_first_compare_NUC_scale_order(scale):
    compare2 = NeedleMan(scale, 10, 1, 1, 1)
    compare2.create_sequence()
    compare2.add_errors()
    compare2.biopython()
    
    compare2.smith()
    return compare2.biopython_score, compare2.biopython_time, compare2.biopython_memory, compare2.biopython_ss, compare2.smith_score, compare2.smith_time, compare2.smith_memory, compare2.smith_ss


def all_aligners_first_compare_NUC_scale_execute():
    fieldnames = [
        "RunID", "Length", "Mutation Rate", "Biopython Score", "Biopython Time", "Biopython Memory", "Biopython_SS", "Smith Score", "Smith Time", "Smith Memory", "Smith SS"]
    
    with open('all_aligners_first_compare_NUC_scale.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

    for scale in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000, 3000, 4000, 5000]:
        for i in range(10):
            current_dict = {
                "RunID": str(uuid.uuid4()),
                "Length": scale,
                "Mutation Rate": 10
            }
            results = all_aligners_first_compare_NUC_scale_order(scale)
            current_dict.update({
                "Biopython Score": results[0], 
                "Biopython Time": results[1], 
                "Biopython Memory": results[2], 
                "Biopython_SS": results[3],  
                "Smith Score": results[4], 
                "Smith Time": results[5], 
                "Smith Memory": results[6], 
                "Smith SS": results[7]
            })
            
            with open('all_aligners_first_compare_NUC_scale.csv', mode='a', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writerow(current_dict)
                print(scale)


