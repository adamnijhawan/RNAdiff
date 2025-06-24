import os
import numpy as np
import glob
import argparse
import mdtraj as md
import barnaba as bb
from barnaba import definitions

def extract_rna_features(cif_file, save_txt_path=None):
    """
    Extract 7 RNA torsion angles (α–ζ, χ) and sugar puckering (P, tm).
    Optionally save per-residue angle values to a .txt file.
    Returns: (N, 9) numpy array in radians
    """
    try:
        # Load torsions
        traj = md.load(cif_file)
        torsion_tensor, mask = bb.backbone_angles(cif_file)  # shape (1, N, 7)
        torsion_matrix = np.squeeze(torsion_tensor, axis=0)  # shape (N, 7)
        torsion_matrix = np.deg2rad(torsion_matrix)          # radians

        # Load puckering
        pucker, mask_s = bb.pucker_angles(cif_file)  # dict with "P", "tm"
        ##tm = pucker["tm"]
        #puckering_matrix = np.stack([P, tm], axis=1)  # shape (N, 2)
        puckering_matrix = np.squeeze(pucker,axis=0)
        # Combine into (N, 9)
        full_torsions = np.concatenate([torsion_matrix, puckering_matrix], axis=1)

        # Write to txt file if path given
        if save_txt_path:
            torsion_labels = ["α", "β", "γ", "δ", "ε", "ζ", "χ", "P", "tm"]
            with open(save_txt_path, "w") as f:
                header = "# Residue " + "".join([f"{label:>10s} " for label in torsion_labels])
                f.write(header + "\n")
                for j in range(full_torsions.shape[0]):
                    line = f"{mask[j]:>10s}" + "".join([f"{val:10.3f} " for val in full_torsions[j]])
                    f.write(line + "\n")
            print(f"[TXT] Saved dihedral angles to {save_txt_path}")

        return full_torsions

    except Exception as e:
        print(f"[ERROR] {os.path.basename(cif_file)}: {e}")
        return None


def process_all_cifs(input_dir, output_dir, save_txt=False):
    os.makedirs(output_dir, exist_ok=True)
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    for cif_file in cif_files:
        base = os.path.splitext(os.path.basename(cif_file))[0]
        txt_path = os.path.join(output_dir, f"{base}_angles.txt") if save_txt else None
        result = extract_rna_features(cif_file, save_txt_path=txt_path)
        if result is not None:
            out_path = os.path.join(output_dir, f"{base}.npy")
            np.save(out_path, result)
            print(f"[OK] Saved {out_path} | shape = {result.shape}")
        else:
            print(f"[SKIP] {cif_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RNA torsions (α–ζ, χ, P, tm) from CIFs and save to .npy and .txt")
    parser.add_argument("--input_dir", required=True, help="Folder containing input .cif RNA structures")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy outputs")
    parser.add_argument("--save_txt", action="store_true", help="Save per-residue torsions to .txt file")

    args = parser.parse_args()
    process_all_cifs(args.input_dir, args.output_dir, save_txt=args.save_txt)
    print(" done")
