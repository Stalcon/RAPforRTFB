# RTFB-RAP-research
Materials screening to determine viable structures of redox active polymers for use in redox targeting flow batteries.

# Computational Workflow
1. 3D structure generation (Rdkit)
2. ML simulations in gas phase (UMA) 
    Gas phase geometry + Internal reorganization energy
3. DFT simulations (SMD)
    Solvated Geometry + External reorganization energy + G in solution (Used to find potential (E))

# Setup

Requires conda or mamba. One command (auto-detects CPU/GPU, installs torch,
deps, xtb, and verifies):

    ./scripts/setup_env.sh

Override backend or env name:

    BACKEND=cpu ./scripts/setup_env.sh
    BACKEND=cuda ./scripts/setup_env.sh
    BACKEND=rocm ./scripts/setup_env.sh
    ./scripts/setup_env.sh myenv

Then activate:

    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate redox

UMA weights need HuggingFace access (https://huggingface.co/facebook/UMA):

    huggingface-cli login

Same code runs either machine via `BACKEND=cpu|cuda|rocm|auto`.

