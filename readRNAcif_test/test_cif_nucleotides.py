from Bio.PDB import MMCIFParser
#test number of nucleotides in RNA structure from CIF; looking into difference between .npy torsions and cif
def count_rna_nucleotides(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("RNA", cif_path)

    count = 0
    for model in structure:
        for chain in model:
            for residue in chain:
                resname = residue.get_resname().strip()
                if resname in {"A", "U", "G", "C"}:
                    count += 1
    return count

# Replace this with your file
cif_file = "/eagle/RNAModeling/RNA_data/train_set/component_6/2il9_A/2il9_A.cif"
print(f"RNA nucleotide count: {count_rna_nucleotides(cif_file)}")
