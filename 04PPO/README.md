
# Robot Estimation Projects: MVU & MLE

This repository contains two simulation-based estimation projects implemented in Python:

- **Project 1: Minimum Variance Unbiased Estimator (MVU)**
- **Project 2: Maximum Likelihood Estimation (MLE) for 2D Localization**

---

## 📁 Project 1: MVU Estimation

### 🔍 Description
Estimates the robot's true 2D position using MVU (sample mean) from noisy measurements.

### ⚙️ Parameters
| Parameter   | Description                           |
|-------------|---------------------------------------|
| `--x`, `--y` | True position of robot               |
| `--n`        | Number of noisy samples per trial    |
| `--trials`   | Number of MVU estimation repetitions |
| `--outdir`   | Output directory for plots           |

### 📊 Output Plots
- `mvu_measurements.png`: Raw noisy samples in 2D
- `mvu_estimate_distribution.png`: Distribution of MVU estimates
- `mvu_variance_plot.png`: Variance trend as number of samples increases

---

## 📁 Project 2: MLE Localization

### 🔍 Description
Estimates the robot's position from noisy distances to known landmarks using MLE.

### ⚙️ Parameters
| Parameter   | Description                             |
|-------------|-----------------------------------------|
| `--x`, `--y` | True robot location                    |
| `--sigma`    | Standard deviation of distance noise   |
| `--outdir`   | Output directory for plots             |

### 📊 Output Plots
- `mle_circle_plot.png`: Landmark distance circles and estimated vs true position

---

## 📈 Result Analysis & Insights

### 🟦 MVU Estimation Results

---

#### 🔹 Baseline (`x=2.0`, `y=3.0`, `n=50`)
- **Measurement Distribution**: The samples are evenly distributed around the true position (2, 3), with some noticeable outliers.
- **Estimate Distribution**: MVU estimates are tightly clustered near the center, forming an elliptical shape.
- **Variance Curve**: As the number of samples increases (`n=10 → 100 → 500`), variance rapidly decreases and converges, demonstrating consistency of the estimator.

---

#### 🔹 Origin Case (`x=0.0`, `y=0.0`, `n=50`)
- **Measurement Distribution**: The samples are symmetrically distributed around the origin with moderate spread.
- **Estimate Distribution**: Estimates are concentrated near (0,0), similar to the baseline case.
- **Variance Curve**: The trend closely follows the baseline, showing stable variance reduction with more samples.

---

#### 🔹 Far Case (`x=5.0`, `y=5.0`, `n=50`)
- **Measurement Distribution**: Samples are more spread out compared to the baseline, centered around (5,5).
- **Estimate Distribution**: Most estimates are near the true value, but outliers increase overall variance slightly.
- **Variance Curve**: Follows a similar pattern to baseline but shows larger variance in the early stages (`n=10~50`), especially along the y-axis.

---

#### 🔹 Small Sample Size (`x=2.0`, `y=3.0`, `n=10`)
- **Measurement Distribution**: Sparse and dispersed samples with noticeable deviations from the center.
- **Estimate Distribution**: Estimates are widely spread with low central density, indicating poor estimation accuracy.
- **Variance Curve**: High variance and slow convergence, highlighting high uncertainty with low sample count.

---

#### 🔹 Medium Sample Size (`x=2.0`, `y=3.0`, `n=100`)
- **Measurement Distribution**: Well-distributed samples centered around (2,3) with few outliers.
- **Estimate Distribution**: Tightly clustered estimates with improved precision over baseline.
- **Variance Curve**: Rapid variance reduction and convergence, demonstrating high reliability.

---

#### 🔹 Large Sample Size (`x=2.0`, `y=3.0`, `n=500`)
- **Measurement Distribution**: Very dense and compact near the true position.
- **Estimate Distribution**: Nearly all estimates converge to a single point, showing high confidence.
- **Variance Curve**: Approaches zero, confirming statistical consistency and efficiency in Gaussian settings.

---

### 🟨 MLE Localization Results

---

#### 🔹 Baseline (`x=5.0`, `y=6.0`, `σ=0.5`)
- **Measurement Circles**: Distance circles from landmarks are balanced and intersect cleanly near the true position.
- **MLE Estimate**: The estimated position closely matches the ground truth, reflecting high accuracy in normal conditions.

---

#### 🔹 Corner Case (`x=1.0`, `y=1.0`, `σ=0.5`)
- **Measurement Circles**: Circles are biased toward one side due to corner placement, reducing overlap clarity.
- **MLE Estimate**: Estimate is near the true position but shows slight distortion due to asymmetric landmark coverage.

---

#### 🔹 Edge Case (`x=9.0`, `y=9.0`, `σ=0.5`)
- **Measurement Circles**: Circles intersect in an unbalanced area due to off-center robot location.
- **MLE Estimate**: Estimate is near the actual location, but reduced overlap increases estimation uncertainty.

---

#### 🔹 Low Noise (`x=5.0`, `y=6.0`, `σ=0.1`)
- **Measurement Circles**: Very tight and precise circles lead to highly accurate intersection.
- **MLE Estimate**: Estimate nearly overlaps with the true location, demonstrating excellent precision under low noise.

---

#### 🔹 High Noise (`x=5.0`, `y=6.0`, `σ=1.0`)
- **Measurement Circles**: Large radius circles result in broad, unclear intersection zones.
- **MLE Estimate**: Estimated position deviates from the true one, clearly showing the negative effect of high noise.
