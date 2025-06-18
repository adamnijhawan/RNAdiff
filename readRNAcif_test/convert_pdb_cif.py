import os
import glob
from Bio.PDB import MMCIFParser, PDBIO

def convert_all_cif_to_pdb(root_dir):
    parser = MMCIFParser(QUIET=True)
    io = PDBIO()

    for cif_file in glob.glob(os.path.join(root_dir, "**", "*.cif"), recursive=True):
        base = os.path.splitext(cif_file)[0]
        pdb_path = base + ".pdb"
        try:
            structure = parser.get_structure(os.path.basename(base), cif_file)
            io.set_structure(structure)
            io.save(pdb_path)
            print(f"[OK] {pdb_path}")
        except Exception as e:
            print(f"[ERROR] {cif_file}: {e}")

if __name__ == "__main__":
    convert_all_cif_to_pdb("/eagle/RNAModeling/RNA_data/train_set/component_6/2il9_A/")
