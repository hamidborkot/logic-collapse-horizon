# Threat Model

This repository accompanies the paper *Logic Collapse Under Compression*. We assume an adversary capable of manipulating model compression techniques (quantization, pruning, or low‑rank adaptation) to induce explanation instability in intrusion detection systems.

## Attacker Goals
- Cause large shifts in feature attribution rankings
- Maintain similar classification accuracy while corrupting explanations

## Defender Goal
Preserve explanation stability under model compression.

## Assumptions
- Attacker cannot alter the dataset directly
- Attacker can manipulate compression configuration

