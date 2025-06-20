import numpy as np
from typing import Union

# Approximate average bond lengths in RNA backbone (in angstroms)
BOND_LENGTHS = {
    ("P", "O5'"): 1.60,
    ("O5'", "C5'"): 1.42,
    ("C5'", "C4'"): 1.53,
    ("C4'", "C3'"): 1.52,
    ("C3'", "O3'"): 1.42,
    ("O3'", "P"): 1.60,  # Next nucleotide
    ("C4'", "O4'"): 1.45,
    ("O4'", "C1'"): 1.42,
    ("C1'", "N9/N1"): 1.47  # Base (purines: N9, pyrimidines: N1)
}

# Approximate average bond angles (in radians)
BOND_ANGLES = {
    ("P", "O5'", "C5'"): 119 * np.pi / 180,
    ("O5'", "C5'", "C4'"): 113 * np.pi / 180,
    ("C5'", "C4'", "C3'"): 104 * np.pi / 180,
    ("C4'", "C3'", "O3'"): 116 * np.pi / 180,
    ("C3'", "O3'", "P"): 119 * np.pi / 180,
    ("C4'", "O4'", "C1'"): 106 * np.pi / 180,
    ("O4'", "C1'", "N9/N1"): 108 * np.pi / 180
}

def unit_vector(vec):
    return vec / np.linalg.norm(vec)

def place_atom(a, b, c, bond_length, bond_angle, torsion_angle):
    bc = unit_vector(c - b)
    n = unit_vector(np.cross(b - a, bc))
    nbc = np.cross(n, bc)
    m = np.stack([bc, nbc, n], axis=-1)

    d = np.array([
        -bond_length * np.cos(bond_angle),
        bond_length * np.cos(torsion_angle) * np.sin(bond_angle),
        bond_length * np.sin(torsion_angle) * np.sin(bond_angle)
    ])

    return c + m @ d

def build_rna_backbone(torsions: np.ndarray) -> np.ndarray:
    """
    Reconstruct 3D RNA backbone coordinates from torsions using NERF.
    Expects torsions in radians, shape (N, 7) where columns are:
    [alpha, beta, gamma, delta, epsilon, zeta, chi]
    """
    coords = []

    # Initialize first three atoms manually (P, O5', C5')
    coords.append(np.array([0.0, 0.0, 0.0]))           # P
    coords.append(np.array([0.0, 0.0, BOND_LENGTHS[("P", "O5'")]]))  # O5'
    coords.append(np.array([0.0, BOND_LENGTHS[("O5'", "C5'")], BOND_LENGTHS[("P", "O5'")]]))  # C5'

    # For each nucleotide
    for i, angles in enumerate(torsions):
        alpha, beta, gamma, delta, epsilon, zeta, chi = angles

        try:
            coords.append(place_atom(coords[-3], coords[-2], coords[-1],
                                     BOND_LENGTHS[("C5'", "C4'")],
                                     BOND_ANGLES[("O5'", "C5'", "C4'")],
                                     alpha))
            coords.append(place_atom(coords[-3], coords[-2], coords[-1],
                                     BOND_LENGTHS[("C4'", "C3'")],
                                     BOND_ANGLES[("C5'", "C4'", "C3'")],
                                     beta))
            coords.append(place_atom(coords[-3], coords[-2], coords[-1],
                                     BOND_LENGTHS[("C3'", "O3'")],
                                     BOND_ANGLES[("C4'", "C3'", "O3'")],
                                     gamma))
            coords.append(place_atom(coords[-3], coords[-2], coords[-1],
                                     BOND_LENGTHS[("O3'", "P")],
                                     BOND_ANGLES[("C3'", "O3'", "P")],
                                     delta))
        except Exception as e:
            print(f"[WARN] Skipping residue {i} due to error: {e}")
            break
        if np.isnan(coords[-1]).any():
            print(f"[DEBUG] NaN introduced at nucleotide {i}.")
    return np.array(coords)

if __name__ == "__main__":
    import sys
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile
    if len(sys.argv) != 2:
        print("Usage: python build_rna.py [input.npy]")
        sys.exit(1)

    torsion_file = sys.argv[1]
    torsions = np.load(torsion_file)

    if torsions.ndim == 3:
        torsions = torsions[0]  # strip batch dim if necessary

    coords = build_rna_backbone(torsions)
    print(coords)
    print("Generated coordinates shape:", coords.shape)
    valid_mask = ~np.isnan(coords).any(axis=1)
    coords = coords[valid_mask]
    atoms = struc.AtomArray(coords.shape[0])
    atoms.coord = coords
    atoms.atom_name[:] = "P"  # Generic atom name (adjust as needed)
    atoms.res_name[:] = "A"
    atoms.res_id[:] = np.arange(1, coords.shape[0] + 1)
    atoms.chain_id[:] = "A"
    atoms.element[:] = "P"

    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write("reconstructed_rna.pdb")
    print("✅ Saved to reconstructed_rna.pdb")