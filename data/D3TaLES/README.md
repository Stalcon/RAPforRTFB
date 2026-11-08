This dataset contains cleaned data for 32,086 molecules marked as "public" in the D3TaLES database at 
the time of this dataset's creation. This dataset excludes molecules with outlier properties. An outlier 
is considered a property value greater than 3 standard deviations from the mean. 

The calculations data were generated through a high-throughput molecular computational workflow 
using density functional theory (DFT) carried out at the (IP-tuned) LC-ωHPBE/Def2SVP level of 
theory via the Gaussian16 (rev A.03) software suite. More details can be found in the following 
article: https://doi.org/10.1039/D3DD00081H. 

The dataset contains 88 possible properties for each molecule. All energy properties are given 
in eV. Properties that begin with `solv_` were calculated in implicit solvent conditions simulating 
acetonitrile (dielectric constant 35.688). Complete documentation for molecular and species properties c
an also be found in the D3TaLES documentation, https://d3tales.as.uky.edu/docs.

Note that care should be taken when evaluating cation 2 and anion 2 properties; these may be highly 
inaccurate. These species with increased charges (+2 or -2) are more likely to form strong interactions 
with individual solvent molecules, meaning that the implicit solvent approximation is less likely to capture
these systems well. 

## Properties (CSV columns)

* `_id`: unique identifier 
* `smiles`: SMILES representation of a molecule
* `source_group`: The source of the original molecular structure
* `groundState_charge`: Charge of the ground state molecule
* `number_of_atoms`: Number of atoms in the molecule
* `molecular_weight`: Molecular weight of the molecule in g/mol
* `sa_score`: Synthetic accessibility (SA) score (https://doi.org/doi:10.1186/1758-2946-1-8) for a SMILES string using the RDKit SA score implementation, sascorer
* `hole_reorganization_energy`: Reorganization energy of cation hole for the molecule
* `electron_reorganization_energy`: Reorganization energy of anion electron for the molecule
* `relaxation_groundState_cation1`: Relaxation energy of the +1 cation from the ground state geometry to the +1 cation geometry
* `relaxation_cation1_groundState`: Relaxation energy of the ground state from the +1 cation geometry to the ground state geometry
* `relaxation_groundState_anion1`: Relaxation energy of the -1 anion from the ground state geometry to the -1 anion geometry
* `relaxation_anion1_groundState`: Relaxation energy of the ground state from the -1 anion geometry to the ground state geometry
* `vertical_ionization_energy`: Vertical ionization energy of the molecule	
* `vertical_electron_affinity`: Vertical electron affinity energy of the molecule
* `adiabatic_ionization_energy`: Adiabatic ionization energy of the molecule
* `adiabatic_electron_affinity`: Adiabatic electron affinity energy of the molecule
* `oxidation_potential`: Oxidation potential of the molecule
* `reduction_potential`: Reduction potential of the molecule
* `omega`: Tuned omega parameter for long ranged corrected functional
* `groundState.charge`: Electronic charge on the groundState
* `cation1.charge`: Electronic charge on the cation1
* `cation2.charge`: Electronic charge on the cation2
* `anion1.charge`: Electronic charge on the anion1
* `anion2.charge`: Electronic charge on the anion2
* `groundState.globular_volume`: Globular volume for groundState in units of cubic Angstroms
* `solv_groundState.globular_volume`:  Globular volume for solv groundState in units of cubic Angstroms
* `cation1.globular_volume`: Globular volume for cation1 in units of cubic Angstroms
* `solv_cation1.globular_volume`:  Globular volume for solv cation1 in units of cubic Angstroms
* `cation2.globular_volume`: Globular volume for cation2 in units of cubic Angstroms
* `solv_cation2.globular_volume`:  Globular volume for solv cation2 in units of cubic Angstroms
* `anion1.globular_volume`: Globular volume for anion1 in units of cubic Angstroms
* `solv_anion1.globular_volume`:  Globular volume for solv anion1 in units of cubic Angstroms
* `anion2.globular_volume`: Globular volume for anion2 in units of cubic Angstroms
* `solv_anion2.globular_volume`:   Globular volume for solv anion2 in units of cubic Angstroms
* `groundState.homo_lumo_gap`: Difference between HOMO and LUMO
* `solv_groundState.homo_lumo_gap`: Difference between HOMO and LUMO
* `cation1.homo_lumo_gap`: Difference between HOMO and LUMO
* `solv_cation1.homo_lumo_gap`: Difference between HOMO and LUMO
* `cation2.homo_lumo_gap`: Difference between HOMO and LUMO
* `solv_cation2.homo_lumo_gap`: Difference between HOMO and LUMO
* `anion1.homo_lumo_gap`: Difference between HOMO and LUMO
* `solv_anion1.homo_lumo_gap`: Difference between HOMO and LUMO
* `anion2.homo_lumo_gap`: Difference between HOMO and LUMO
* `solv_anion2.homo_lumo_gap`: Difference between HOMO and LUMO
* `groundState.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `solv_groundState.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `cation1.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `solv_cation1.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `cation2.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `solv_cation2.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `anion1.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `solv_anion1.dipole_moment`:  The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `anion2.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `solv_anion2.dipole_moment`: The X, Y, and Z dipole components of electric dipole moment. Units: Debye
* `groundState.solvation_energy`: Difference between gas phase energy and solvated energy
* `solv_groundState.solvation_energy`: Difference between gas phase energy and solvated energy
* `cation1.solvation_energy`: Difference between gas phase energy and solvated energy
* `solv_cation1.solvation_energy`: Difference between gas phase energy and solvated energy
* `cation2.solvation_energy`: Difference between gas phase energy and solvated energy
* `solv_cation2.solvation_energy`: Difference between gas phase energy and solvated energy
* `anion1.solvation_energy`: Difference between gas phase energy and solvated energy
* `solv_anion1.solvation_energy`: Difference between gas phase energy and solvated energy
* `anion2.solvation_energy`: Difference between gas phase energy and solvated energy
* `solv_anion2.solvation_energy`: Difference between gas phase energy and solvated energy
* `groundState.homo`: Highest occupied molecular orbital. Units: eV
* `solv_groundState.homo`: Highest occupied molecular orbital. Units: eV
* `cation1.homo`: Highest occupied molecular orbital. Units: eV
* `solv_cation1.homo`: Highest occupied molecular orbital. Units: eV
* `cation2.homo`: Highest occupied molecular orbital. Units: eV
* `solv_cation2.homo`: Highest occupied molecular orbital. Units: eV
* `anion1.homo`: Highest occupied molecular orbital. Units: eV
* `solv_anion1.homo`: Highest occupied molecular orbital. Units: eV
* `anion2.homo`: Highest occupied molecular orbital. Units: eV
* `solv_anion2.homo`: Highest occupied molecular orbital. Units: eV
* `groundState.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `solv_groundState.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `cation1.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `solv_cation1.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `cation2.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `solv_cation2.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `anion1.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `solv_anion1.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `anion2.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `solv_anion2.lumo`: Lowest unoccupied molecular orbital. Units: eV
* `groundState.singlet_states`: Species singlet excited states
* `cation1.singlet_states`: Species singlet excited states
* `cation2.singlet_states`: Species singlet excited states
* `anion1.singlet_states`: Species singlet excited states
* `anion2.singlet_states`: Species singlet excited states
* `groundState.triplet_states`: Species triplet excited states
* `cation1.triplet_states`: Species triplet excited states
* `cation2.triplet_states`: Species triplet excited states
* `anion1.triplet_states`: Species triplet excited states
* `anion2.triplet_states`: Species triplet excited states
