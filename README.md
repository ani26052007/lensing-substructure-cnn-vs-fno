# CNN and Neural Operator Approaches for Gravitational Lens Substructure Classification

End-to-end deep learning study on strong gravitational lensing image classification, comparing a transfer learning CNN baseline (ConvNeXt V2) against a Fourier Neural Operator approach with physics-motivated input engineering.

---

## Problem Statement

Strong gravitational lensing occurs when a massive foreground object bends light from a background source, producing characteristic Einstein ring patterns. The substructure of the foreground lens — whether it contains dark matter subhalos, vortex perturbations, or is a smooth distribution — leaves distinct but subtle imprints on the ring morphology.

The goal is to classify 150×150 grayscale lensing simulations into three categories:

| Class | Physical interpretation |
|-------|------------------------|
| `no_sub` | Smooth mass distribution, produces a clean, symmetric Einstein ring |
| `subhalo` | Dark matter subhalo, causes localized arc distortions from the mass concentration |
| `vortex` | Vortex substructure, produces distributed, asymmetric ring perturbations |

**Primary evaluation metric:** ROC AUC (one-vs-rest macro average)

---

## Dataset

| Split | Samples | Per class |
|-------|---------|-----------|
| Training | 30,000 | 10,000 |
| Validation | 7,500 | 2,500 |

Images are perfectly balanced, single-channel, min-max normalized to [0, 1].

---

## Tasks Covered

### Task 1: ConvNeXt V2 Classifier

A transfer learning approach using a pretrained ConvNeXt V2 Tiny backbone with physics-motivated 3-channel input engineering and four-stage discriminative fine-tuning.

**Notebook:** `task-1.ipynb`  
**README:** `README_Task_1.md`

**Key design choices:**
- Physics-motivated 3-channel input (raw + gradient magnitude + Laplacian)
- ConvNeXt V2 Tiny pretrained on ImageNet-1k via FCMAE
- Four-stage discriminative fine-tuning with frozen backbone warm-up
- Test-Time Augmentation over 8 geometric transforms
- Grad-CAM validation confirming physically meaningful attention

**Result:** Val Macro AUC 0.9819, Val Accuracy 91.19%

---

### Task 2: Fourier Neural Operator Classifier

A neural operator approach replacing the standard CNN feature extractor with Fourier Neural Operator (FNO) blocks that learn in frequency space, combined with a polar-domain input representation designed specifically for FNO's spectral convolutions.

**Notebooks:** `fno.ipynb`, `channel_analysis_Task_4.ipynb`  
**README:** `README_Task_2_FNO.md`

**Key design choices:**
- 4-channel polar-domain input (raw + polar transform + angular gradient + radial deviation)
- FNO2d backbone with SpectralConv2d blocks (global receptive field from the first layer)
- Trained from scratch — no pretrained FNO weights available for this domain
- Per-class threshold grid search during TTA for accuracy optimization
- Extensive channel ablation study

**Result:** Val Macro AUC 0.9698, Val Accuracy 87.59%

---

## Architecture Comparison

| Aspect | Task 1 (ConvNeXt V2) | Task 2 (FNO) |
|--------|----------------------|----------------|
| Backbone | ConvNeXt V2 Tiny | FNO2d (4 blocks, width=64, modes=20) |
| Input channels | 3 (raw + grad + Laplacian) | 4 (raw + polar + angular grad + radial dev) |
| Feature extraction | Local (depthwise conv) → stacked for global context | Global (FFT → spectral multiply → IFFT in a single layer) |
| Pretraining | ImageNet-1k (FCMAE, ~28M params) | None, trained from 30k lensing images |
| Training strategy | 4-stage discriminative fine-tuning | Two-phase: full training → LR-reduced resume |
| Val Macro AUC | 0.9819 | 0.9698 |

---

## Key Technical Decisions

### Physics-Informed Input Engineering

Neither task uses raw single-channel input. Both leverage domain knowledge about lensing physics to construct richer multi-channel representations:

- **Task 1:** Differential operators (gradient magnitude, Laplacian) computed at native 150×150 resolution before upsampling — the operators act on original image structure, not interpolated pixels
- **Task 2:** Polar coordinate transformation followed by angular and radial symmetry-breaking channels, specifically designed to map lensing physics into frequency-domain-friendly representations

### CNN vs Neural Operator: Theoretical Motivation

The Einstein ring is a global, ring-shaped structure. Standard CNNs require many stacked layers to accumulate global receptive fields, while a single FNO spectral convolution sees the entire image. The FNO's 20 retained frequency modes (20/150 ≈ 13% of the spectrum) target the arc morphology frequency band, roughly 7–75 pixel wavelengths.

The lack of pretrained FNO weights means the neural operator must learn all representations from 30k samples, whereas ConvNeXt V2 brings rich low-level feature extractors from ImageNet. This pretraining advantage is the dominant factor in the performance gap between the two tasks.

### Validation of Physical Correctness

Both models are validated with Grad-CAM to ensure learned features correspond to physically meaningful regions:

- `no_sub` → attention at lens center (smooth ring symmetry)
- `subhalo` → attention on localized arc perturbation (mass concentration)
- `vortex` → attention on asymmetric arc region (distributed perturbation)

Models that exploit background noise or image corners would be flagged and retrained.

---

## References

- Li et al., "Fourier Neural Operator for Parametric Partial Differential Equations", ICLR 2021
- Kovachki et al., "Neural Operator: Learning Maps Between Function Spaces", JMLR 2023
- Woo et al., "ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders", CVPR 2023
- Ojha et al., "LensPINN", NeurIPS ML4PS 2024
- Marr & Hildreth, "Theory of Edge Detection", Proc. Royal Society London, 1980
- Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks", ICCV 2017
