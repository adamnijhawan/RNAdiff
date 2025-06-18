import os
import numpy as np
import glob
import argparse
import mdtraj as md
import barnaba as bb
#from barnaba.analysis import backbone_angles
def extract_rna_features(cif_file, include_pucker=False):
    """
    Extract 7 RNA torsion angles (α–ζ, χ) and optionally sugar puckering (P, tm).
    Returns: (N, 7) or (N, 9) numpy array in radians
    """
    try:
        print('loading traj')
        traj = md.load(cif_file)  # barnaba autodetects mmCIF
        #traj[0].save_pdb("output.pdb")
        print('loaded traj')
        #print(traj)
        torsions = bb.backbone_angles(cif_file)
        torsion_tensor, mask = bb.backbone_angles(cif_file)

# Assume single-frame (static structure)
        torsion_matrix = np.squeeze(torsion_tensor, axis=0)  # shape: (135, 7)
        torsion_matrix = np.deg2rad(torsion_matrix)          # convert to radians

        #print(torsions)'''
        '''keys = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        #torsions = bb.backbone_angles(traj)

        if not isinstance(torsions, dict):
            print(f"[ERROR] Unexpected output from backbone_angles for {os.path.basename(cif_file)}: type = {type(torsions)}")
            return None

        required_keys = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "chi"]
        if not all(k in torsions for k in required_keys):
            print(f"[ERROR] Missing torsion keys in {os.path.basename(cif_file)}")
            return None
        '''
        #torsion_rad = [np.deg2rad(torsions[k]) for k in required_keys]


        #torsion_rad = [np.deg2rad(torsions[k]) for k in keys]
        #torsion_matrix = np.stack(torsion_rad, axis=1)  # shape (N, 7)

        if include_pucker:
            pucker = bb.pucker_angles(cif_file)
            P = np.deg2rad(pucker["P"])      # pseudorotation angle
            tm = pucker["tm"]                # amplitude
            extra = np.stack([P, tm], axis=1)  # shape (N, 2)
            return np.concatenate([torsion_matrix, extra], axis=1)  # shape (N, 9)
        else:
            return torsion_matrix

    except Exception as e:
        print(f"[ERROR] {os.path.basename(cif_file)}: {e}")
        return None

def process_all_cifs(input_dir, output_dir, include_pucker=False):
    os.makedirs(output_dir, exist_ok=True)
    cif_files = glob.glob(os.path.join(input_dir, "*.cif"))
    print(cif_files)
    for cif_file in cif_files:
        print(cif_file)
        result = extract_rna_features(cif_file, include_pucker=include_pucker)
        if result is not None:
            base = os.path.splitext(os.path.basename(cif_file))[0]
            out_path = os.path.join(output_dir, f"{base}.npy")
            np.save(out_path, result)
            print(f"[OK] Saved {out_path} | shape = {result.shape}")
        else:
            print(f"[SKIP] {cif_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract RNA torsions (and optionally puckering) from CIFs to .npy")
    parser.add_argument("--input_dir", required=True, help="Folder containing input .cif RNA structures")
    parser.add_argument("--output_dir", required=True, help="Where to save .npy outputs")
    parser.add_argument("--include_pucker", action="store_true", help="Include sugar puckering (P, tm) as extra 2 columns")
    args = parser.parse_args()

    process_all_cifs(args.input_dir, args.output_dir, include_pucker=args.include_pucker)
    print('done')
