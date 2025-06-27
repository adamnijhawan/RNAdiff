import numpy as np
from typing import Union

# RNA backbone bond lengths (in Å)
BOND_LENGTHS = {
    ("P", "O5'"): 1.60,
    ("O5'", "C5'"): 1.42,
    ("C5'", "C4'"): 1.53,
    ("C4'", "C3'"): 1.52,
    ("C3'", "O3'"): 1.42,
    ("O3'", "P"): 1.60,
}
#testing
# Bond angles (in radians)
BOND_ANGLES = {
    ("P", "O5'", "C5'"): np.deg2rad(119),
    ("O5'", "C5'", "C4'"): np.deg2rad(113),
    ("C5'", "C4'", "C3'"): np.deg2rad(104),
    ("C4'", "C3'", "O3'"): np.deg2rad(116),
    ("C3'", "O3'", "P"): np.deg2rad(119),
    ("O3'", "P", "O5'"): np.deg2rad(119),
}
print('testing')
# Backbone atoms in order
BACKBONE_ORDER = ["P", "O5'", "C5'", "C4'", "C3'", "O3'"]

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

def build_rna_backbone(torsions: np.ndarray) -> tuple:
    """
    Reconstruct 3D RNA backbone atoms from 7 torsion angles.
    """
    coords = []
    names = []

    # Seed: initial 3 atoms (P, O5', C5')
    coords.append(np.array([0.0, 0.0, 0.0]))                            # P
    coords.append(np.array([0.0, 0.0, BOND_LENGTHS[("P", "O5'")]]))     # O5'
    coords.append(np.array([0.0, BOND_LENGTHS[("O5'", "C5'")], BOND_LENGTHS[("P", "O5'")]]))  # C5'

    names += ["P", "O5'", "C5'"]

    for i, (alpha, beta, gamma, delta, epsilon, zeta, chi) in enumerate(torsions):
        try:
            # Place C4'
            C5p, prev2, prev1 = coords[-3], coords[-2], coords[-1]
            C4p = place_atom(C5p, prev2, prev1,
                         BOND_LENGTHS[("C5'", "C4'")],
                         BOND_ANGLES[("O5'", "C5'", "C4'")],
                         alpha)
            coords.append(C4p)
            names.append("C4'")

            # Place C3'
            C3p = place_atom(prev2, prev1, C4p,
                         BOND_LENGTHS[("C4'", "C3'")],
                         BOND_ANGLES[("C5'", "C4'", "C3'")],
                         beta)
            coords.append(C3p)
            names.append("C3'")

            # Place O3'
            O3p = place_atom(prev1, C4p, C3p,
                         BOND_LENGTHS[("C3'", "O3'")],
                         BOND_ANGLES[("C4'", "C3'", "O3'")],
                         gamma)
            coords.append(O3p)
            names.append("O3'")

            # Place next P
            P_next = place_atom(C4p, C3p, O3p,
                            BOND_LENGTHS[("O3'", "P")],
                            BOND_ANGLES[("C3'", "O3'", "P")],
                            delta)
            coords.append(P_next)
            names.append("P")

            # Place O5'
            O5p_next = place_atom(C3p, O3p, P_next,
                              BOND_LENGTHS[("P", "O5'")],
                              BOND_ANGLES[("O3'", "P", "O5'")],
                              epsilon)
            coords.append(O5p_next)
            names.append("O5'")

            # Place C5'
            C5p_next = place_atom(O3p, P_next, O5p_next,
                              BOND_LENGTHS[("O5'", "C5'")],
                              BOND_ANGLES[("P", "O5'", "C5'")],
                              zeta)
            coords.append(C5p_next)
            names.append("C5'")
        except Exception as e:
            print(f"[WARN] Error at nucleotide {i}: {e}")
            break

    return np.array(coords), names

if __name__ == "__main__":
    import sys
    import biotite.structure as struc
    from biotite.structure.io.pdb import PDBFile

    if len(sys.argv) != 2:
        print("Usage: python build_rna.py [torsions.npy]")
        sys.exit(1)

    torsion_file = sys.argv[1]
    torsions = np.load(torsion_file)

    if torsions.ndim == 3:
        torsions = torsions[0]

    coords, atom_names = build_rna_backbone(torsions)
    valid_mask = ~np.isnan(coords).any(axis=1)
    coords = coords[valid_mask]
    atom_names = [name for i, name in enumerate(atom_names) if valid_mask[i]]

    for a, n in zip(coords, atom_names):
        print(f"{n:4} | {a}")
    atoms = struc.AtomArray(coords.shape[0])
    atoms.coord = coords
    atoms.atom_name[:] = atom_names
    atoms.res_name[:] = "A"
    atoms.res_id[:] = np.arange(1, coords.shape[0] + 1)
    atoms.chain_id[:] = "A"
    atoms.element[:] = [name[0] for name in atom_names]  # crude element guess

    pdb = PDBFile()
    pdb.set_structure(atoms)
    pdb.write("reconstructed_rna.pdb")
    print("Saved to reconstructed_rna.pdb")
