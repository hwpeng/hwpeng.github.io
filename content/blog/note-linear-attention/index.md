---
title: "Note: Linear Attention"
date: '2026-01-28'
---


These [notes](Linear_Attention.pdf) introduce linear attention and its variants. We cover: 
- How linear attention compresses key-value history into a fixed-size state, reducing complexity from $\mathcal{O}(L^2)$ to $\mathcal{O}(L)$;
- The Delta Rule and gated variants for improved memory fidelity; 
- Chunkwise algorithms that enable parallel training via the WY representation and UT transformation. Complete derivations and algorithms are provided.