# Self-Organized Criticality in the Bak–Tang–Wiesenfeld Sandpile

**Author:** Kanishk Tak  
**Department:** Chemical Engineering, IIT Delhi  
**Course:** CLL 798 — Complexity Sciences  
**Semester:** Spring 2026

## Project Overview

This project applies complexity science to the **Bak–Tang–Wiesenfeld (BTW) sandpile cellular automaton**, the paradigmatic model for self-organized criticality (SOC). Grains of sand dropped onto an L×L lattice drive the system to a critical state where avalanche sizes, areas, and durations all follow power-law distributions—without any parameter fine-tuning.

## Repository Structure

```
.
├── sandpile_main.tex         # LaTeX source (Physica A format, 28 pages)
├── sandpile_main.pdf         # Compiled manuscript
├── sandpile_refs.bib         # BibTeX references (22 entries)
├── btw_simulation.py         # Full simulation code (real data, no placeholders)
└── README.md                 # This file
```

## Key Results

| Rubric Item | Result |
|---|---|
| **4a** Why complexity science? | BTW sandpile self-tunes to criticality from any initial state; avalanche statistics match neural, seismic, and wildfire data |
| **4b** Noise, Avalanche, Connectivity | Random grain drops (noise) → toppling cascades (avalanches) → threshold z_c controls branching ratio (connectivity) |
| **4c** Simulation | Python/NumPy vectorised BTW with open boundaries; tracks size, area, and duration per avalanche |
| **4d** Frequency distribution vs connectivity | z_c → z̄* → branching ratio μ → τ̂_s; sensitivity heatmap τ̂_s(z_c, L) confirms systematic variation |
| **4e** SOC assessment | Scale-free P(s), P(a), P(T); Abelian structure (Dhar's theorem); FSS: s_max ~ L^{D_f}; 1/f^α noise detected |

## Measured Exponents

| Observable | Exponent | Literature |
|---|---|---|
| Avalanche size | τ̂_s ≈ 1.10 ± 0.02 | 1.05–1.15 |
| Avalanche area | τ̂_a ≈ 1.22 ± 0.03 | 1.20–1.25 |
| Avalanche duration | τ̂_T ≈ 1.49 ± 0.04 | 1.45–1.55 |

## How to Run the Simulation

### Quick test (~2 minutes)
```bash
python3 btw_simulation.py --fast
```

### Full run — reproduces all manuscript figures (~15 minutes)
```bash
python3 btw_simulation.py
```

Figures are saved to `figs/` and numerical results to `results.json`.

### Compile PDF from LaTeX source
```bash
pdflatex sandpile_main.tex
bibtex sandpile_main
pdflatex sandpile_main.tex
pdflatex sandpile_main.tex
```

## Dependencies

```
Python >= 3.9
numpy
matplotlib
scipy
```

Install with:
```bash
pip install numpy matplotlib scipy
```

## Model Description

The BTW sandpile evolves on an L×L lattice with **open boundary conditions**:

1. **Drive**: Drop one grain at a uniformly random site — the stochastic *noise*
2. **Topple**: Any site with height z_{i,j} ≥ z_c loses z_c grains and each of its 4 von Neumann neighbours gains 1 grain
3. **Dissipate**: Grains leaving the lattice boundary are lost (energy sink sustaining the non-equilibrium steady state)
4. **Measure**: Record the total topplings (size s), distinct toppled sites (area a), and toppling rounds (duration T) per grain drop

The critical threshold **z_c is the connectivity parameter**: it controls the mean branching ratio μ = z · P(z = z_c − 1) of the toppling cascade. The standard BTW model (z_c = 4, von Neumann neighbourhood) sits at μ ≈ 1 in 2D, the SOC critical point.

## Statistical Methods

- **MLE power-law fitting**: Clauset, Shalizi & Newman (2009)
- **Truncated power-law MLE**: Nelder-Mead optimisation of full log-likelihood L(τ, λ)
- **Likelihood-ratio test**: Pure PL vs TPL (Wilks' theorem, χ²₁)
- **Finite-size scaling**: s_max ~ L^{D_f} across L ∈ {20, 30, 40, 50, 60}
- **AIC/BIC**: Model selection for functional forms

## References

- Bak, P., Tang, C. & Wiesenfeld, K. (1987). *Phys. Rev. Lett.* 59, 381.
- Dhar, D. (1990). *Phys. Rev. Lett.* 64, 1613 (Abelian property).
- Priezzhev, V. B. (1994). *J. Stat. Phys.* 74, 955 (exact height probabilities).
- Pruessner, G. (2012). *Self-Organised Criticality*. Cambridge University Press.
- Clauset, A. et al. (2009). *SIAM Review* 51, 661.

## AI Transparency

AI tools (Claude, Anthropic) were used for code generation and manuscript drafting. All scientific decisions — model choice, statistical methodology, interpretation of results, identification of z_c as the connectivity parameter, and retraction of non-results — were made by the author. See Appendix B of the manuscript for the full prompt log.
