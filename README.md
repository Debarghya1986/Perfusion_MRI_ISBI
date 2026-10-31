# Trained Spatiotemporal Regularization for Perfusion MRI Deconvolution

**RED-CCD: Nonnegative model-based reconstruction using a trained spatiotemporal regularizer**

Research implementation accompanying the manuscript:

**Debarghya Nandi, Trishit Mukherjee, and Kunal N. Chaudhury**  
Indian Institute of Science, Bengaluru, India.

**Manuscript status:** Under review.

## Overview

Dynamic susceptibility contrast (DSC) MRI estimates cerebral perfusion by deconvolving measured tissue concentration curves with an arterial input function (AIF). This inverse problem is ill-conditioned and sensitive to measurement noise.

We propose RED-CCD, a model-based reconstruction method that combines:

- An explicit AIF-based forward model.
- A trained, noise-conditioned spatiotemporal denoiser.
- A smooth, strongly convex regularizer derived from the denoising residual.
- Nonnegative projected-gradient reconstruction with convergence guarantees.

The denoiser is trained independently of the AIF-based forward model, allowing reconstruction under different AIF conditions without retraining.

## Method

The discrete forward model is

\[
c = Ah + n,
\]

where \(A\) represents temporal convolution with the AIF, \(h\) is the unknown tissue response, and \(c\) is the measured tissue concentration.

Our reconstruction solves

\[
\min_{h \geq 0}
\frac{1}{2}\|Ah-c\|_2^2+\lambda\gamma g(h),
\]

where the trained denoiser satisfies

\[
D_\sigma(h)=h-\gamma\nabla g(h).
\]

The corresponding algorithm uses projected gradient descent, enforcing nonnegative tissue responses.

## Synthetic Dataset

The synthetic DSC-MRI dataset includes:

- 512 training, 128 validation, and 120 test cases.
- Balanced bi-exponential and pharmacokinetic tissue-response families.
- Gray matter, white matter, and ischemic core regions.
- Spatial dimensions of 66 x 77 pixels.
- 57 temporal samples acquired at 1-second intervals.
- Signal-domain Gaussian noise followed by conversion to concentration.
- Nominal, delayed, and broad AIF evaluation conditions.

The complete dataset is not stored in Git. The generation scripts and configurations are provided.

## Repository Contents

| Component | Description |
|---|---|
| `core/` | CCD architecture, convolutions, and trainable splines |
| `training/` | Denoiser training implementation |
| `spline01_model.py` | Final noise-conditioned CCD model |
| `25_projected_red_aif.py` | Nonnegative projected RED solver |
| `32_train_spline01.py` | Final denoiser training entry point |
| `33_test_spline01_nonnegative.py` | Final RED-CCD evaluation |
| `39_tune_spline01_nonnegative_nominal.py` | Validation-only hyperparameter tuning |
| `v3/` | Synthetic dataset and evaluation utilities |
| `osvd/` | oSVD baseline |
| `rotkopf_pinn/` | PINN baseline |
| `spinned_style/` | SPINNED-style baseline |
| `frindel_spatiotemporal.py` | Frindel-style baseline |
| `34_report_spline01_results.py` | Reconstruction results reporting |
| `35_paper_figure_io.py` | Figure data-loading utilities |
| `36_plot_nominal_three_baselines_spline.py` | Nominal-AIF tissue-response figures |
| `37_plot_aif_ccd_vs_spinned_spline.py` | AIF-generalization figures |
| `38_plot_final_perfusion_maps_spline.py` | Perfusion-map figures |

## Installation

Python with PyTorch is required. CUDA is recommended for reconstruction experiments.

```bash
git clone https://github.com/Debarghya1986/Perfusion_MRI_ISBI.git
cd Perfusion_MRI_ISBI

pip install -r requirements.txt
```

Install a PyTorch build appropriate for your hardware if the default installation is unsuitable.

## Generate Synthetic Data

```bash
python 01_generate_synthetic_v3.py --config config_v3.yml
python 02_verify_synthetic_v3.py --config config_v3.yml
```

The generator uses the dataset parameters and random seed specified in the configuration.

## RED-CCD Reconstruction

The final trained CCD checkpoint is included at:

`checkpoints_ccd3d_spline01/ccd3d_spline01_best_inference.pt`

For a single-case reconstruction test:

```bash
python 33_test_spline01_nonnegative.py \
  --config config_v3_spline01_tuned.yml \
  --conditions baseline \
  --max-cases 1 \
  --batch-size 1
```

For full evaluation, generate all test cases and the paired delayed/broad AIF datasets first.

The final validation-selected parameters are:

- Regularization parameter: 1.6
- CCD strength: 1/255
- Maximum iterations: 5000
- Relative convergence tolerance: 1e-5

## Reconstruction Constraints

The RED variants use different reconstruction constraints,
consistent with the experiments reported in the manuscript.

| Method | Reconstruction | Constraint |
|---|---|---|
| RED-CCD (ours) | Projected RED with trained CCD prior | Nonnegative |
| RED-TV | Projected RED with TV prior | Nonnegative |
| RED-BM4D | Standard RED with BM4D prior | Unconstrained |

- RED-CCD uses the final spline01 checkpoint and projected solver.
- RED-TV uses the projected solver in `25_projected_red_aif.py`.
- RED-BM4D uses the unconstrained implementation in
  `04_benchmark_red_priors_initC.py`.
- Nonnegative clipping for perfusion-map evaluation is distinct
  from enforcing nonnegativity during reconstruction.

The optional projected-BM4D pathway in
`25_projected_red_aif.py` is not the RED-BM4D variant
reported in the final nominal-AIF comparison.

## Baseline Implementations

The repository includes the following comparison methods:

- RED-TV: Total-variation regularization with nonnegative projected RED.
- RED-BM4D: CUDA BM4D denoising with unconstrained RED reconstruction.
- Frindel-ST: Spatiotemporal deconvolution.
- oSVD: Oscillatory singular value decomposition.
- PINN: Physics-informed neural network.
- SPINNED-style: Learning-based perfusion deconvolution.

The CUDA BM4D implementation is provided in
`third_party/bm4d_cuda.py`, with optional dependencies in
`requirements-bm4d-cuda.txt`.

The complete multi-method benchmark has not yet been
independently rerun from a fresh public checkout.

## Evaluation

We report:

- CBF absolute relative error (%)
- CBV absolute relative error (%)
- MTT absolute relative error (%)
- Tissue-response normalized root-mean-square error (h-NRMSE)

The manuscript compares RED-CCD with RED-TV, RED-BM4D, Frindel-ST, oSVD, PINN, and a SPINNED-style network.

All synthetic test cases are held out from denoiser training and hyperparameter selection.

## Reproducibility Notes

The final trained RED-CCD checkpoint is provided for inference.

The release preserves the trained spline01 architecture and the projected nonnegative reconstruction algorithm. Synthetic data generation, validation, baseline wrappers and evaluation scripts, and figure-generation code are included. Some baseline pathways require additional dependencies or verification, as noted below.

This is a simulation study. Validation on real DSC-MRI data remains future work.

## Manuscript

**Trained Spatiotemporal Regularization for Perfusion MRI Deconvolution**

Debarghya Nandi, Trishit Mukherjee, and Kunal N. Chaudhury.

*Under review. The manuscript will be made available following the appropriate publication or preprint release.*

## Citation

If you use this work, please cite the accompanying manuscript and research software.

See `CITATION.cff` for software citation metadata.
