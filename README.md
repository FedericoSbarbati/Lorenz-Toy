# Project: Embedding for Spatiotemporal Dynamics Reconstruction

## Overview
This project is inspired by the supplementary appendix of the research paper:

**Raut, R. V., Rosenthal, Z. P., Wang, X., Miao, H., Zhang, Z., Lee, J.-M., Raichle, M. E., Bauer, A. Q., Brunton, S. L., Brunton, B. W., & Kutz, J. N. (2023). Arousal as a universal embedding for spatiotemporal brain dynamics. _bioRxiv_.**

Our implementation evaluates the ability of variational autoencoders (VAEs) to reconstruct spatiotemporal data and explore PCA-based representations for dimensionality reduction. This repository focuses on analyzing the reconstruction of a Stochastic Lorenz attractor using Delay Coordinate Embedding, aiming to replicate and enhance the findings from the referenced paper.

---

## Features

### 1. **Encoder and Decoder Training**
- Configurations for model training are stored in `.json` files.
- Training supports:
  - Latent dimensionality adjustments.
  - Dynamic scheduling strategies for optimization.
  - Beta-VAE techniques for Kullback-Leibler divergence annealing.

### 2. **Analysis Pipelines**
- Statistical evaluation includes:
  - R², MSE, and MaxSE metrics for reconstruction performance.
  - PCA-based variance and time-series reconstructions.
- Results are saved as `.csv` files for seamless visualization.

### 3. **Visualization**
- Rich plotting functionalities:
  - Comparative bar plots for reconstruction metrics.
  - Line plots of PCA-explained variance and reconstruction R² across models.
  - Detailed per-model analysis for both encoders and decoders.

---

## Repository Structure
```plaintext
├── Encoder Models/         # Saved weights for encoder networks
├── Decoder Models/         # Saved weights for decoder networks
├── Encoder Analysis Data/  # CSV files with encoder analysis results
├── Decoder Analysis Data/  # CSV files with decoder analysis results
├── Tools/                  # Utility functions for training, analysis, and plotting
├── Notebooks/              # Jupyter notebooks for training and analyzing models
└── README.md               # Project introduction
```

---
## Usage Instructions

### Training Encoders
1. Open the `trainEncoder.ipynb` notebook to define the encoder configuration and train the model.
2. After training, use the `valEncoder.ipynb` notebook to:
   - Evaluate the encoder's performance.
   - Analyze reconstruction capabilities and PCA-based dimensionality reduction.
   - Save the analysis results.

### Training Decoders
1. Open the `trainDecoder.ipynb` notebook to:
   - Define the decoder configuration.
   - Provide the associated encoder’s name for integration.
   - Train the decoder using the specified configuration.
2. After training, use the `valDecoder.ipynb` notebook to:
   - Evaluate the decoder's reconstruction performance.
   - Analyze metrics related to spatiotemporal data reconstruction.
   - Save the results for further comparison.

### Running Analysis
1. For single-model analysis:
   - Use the validation notebooks (`valEncoder.ipynb` or `valDecoder.ipynb`) to analyze individual models.
   - Save the analysis results in dedicated `.csv` files.
2. For comparative analysis:
   - Use the analysis notebook to:
     - Load and aggregate analysis data from all trained models.
     - Compare statistical metrics across models and noise levels.
     - Generate visualizations for detailed insights into performance.



---

## Dependencies

- Python 3.8+
- PyTorch
- NumPy
- Matplotlib
- Pandas

Install all dependencies with:
```bash
pip install -r requirements.txt
```

---

## References
This project builds upon the methodology and findings described in the following paper:

**Raut, R. V., Rosenthal, Z. P., Wang, X., Miao, H., Zhang, Z., Lee, J.-M., Raichle, M. E., Bauer, A. Q., Brunton, S. L., Brunton, B. W., & Kutz, J. N. (2023). Arousal as a universal embedding for spatiotemporal brain dynamics. _bioRxiv_.**

Supplementary materials from the appendix served as a foundation for architectural choices and evaluation metrics used in this repository. You can access the full paper [here](https://doi.org/10.1101/2023.11.06.565918).

---

## Contributing
Contributions are welcome! If you’d like to add features, improve documentation, or optimize training methods, please submit a pull request.

---

## Contact
For questions, feedback, or collaborations, reach out to the repository maintainer via GitHub Issues.

---

## License
This repository is released under the MIT License.

