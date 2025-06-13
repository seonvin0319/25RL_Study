
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

### 🔹 MVU Estimation: Baseline (`x=2.0`, `y=3.0`, `n=50`)
- **Measurement Distribution**: 측정값들은 중심인 (2,3)을 기준으로 고르게 퍼져 있으며, 일부 외곽 outlier가 존재함.
- **Estimate Distribution**: MVU 추정값은 중심에 밀집되어 있고, 비교적 타원형 분포를 가짐. 평균적으로 정확하게 중심 근처에 모임.
- **Variance Curve**: `n=10 → 100 → 500`로 갈수록 x, y 축 분산이 빠르게 감소하며 수렴. MVU 추정기의 일치성(consistency)이 잘 드러남.

---

### 🔹 MVU Estimation: Origin (`x=0.0`, `y=0.0`, `n=50`)
- **Measurement Distribution**: 좌표 원점 주변에서 비교적 대칭적인 분포를 보이며, 전반적으로 퍼짐은 적절함.
- **Estimate Distribution**: 중심이 0에 가까워서 시각적으로도 추정 분포가 원점 주변에 잘 몰림. 분산은 baseline과 유사한 수준.
- **Variance Curve**: baseline 실험과 유사한 경향성을 가지며, 분산은 샘플 수에 따라 안정적으로 감소.

---

### 🔹 MVU Estimation: Far (`x=5.0`, `y=5.0`, `n=50`)
- **Measurement Distribution**: 전반적으로 (5,5) 주변에 모여 있으나, baseline보다 좀 더 넓게 퍼지는 경향이 있음.
- **Estimate Distribution**: 중심 추정은 잘 맞지만, 일부 외곽 추정값이 보여서 분산은 약간 더 큰 듯한 경향.
- **Variance Curve**: baseline, origin과 유사한 패턴을 보이나, 초기 구간(n=10~50)에서 y축 분산이 상대적으로 더 큼.

---

### 🔹 MVU Estimation: Small Sample (`x=2.0`, `y=3.0`, `n=10`)
- **Measurement Distribution**: 샘플 수가 적어 외곽에 위치한 값들이 눈에 띄며, 측정값의 중심 집중도가 낮음.
- **Estimate Distribution**: MVU 추정값이 퍼져 있고 타원 형태의 밀집도가 낮으며, 불균형한 추정 분포가 드러남.
- **Variance Curve**: 분산 값이 높고, 감소 속도도 느림. 작은 n에서는 MVU 추정기의 불확실성이 크게 나타남.

---

### 🔹 MVU Estimation: Medium Sample (`x=2.0`, `y=3.0`, `n=100`)
- **Measurement Distribution**: (2,3)을 기준으로 고르게 퍼진 분포를 보이며, outlier는 거의 없음.
- **Estimate Distribution**: 추정값들이 중심에 뚜렷하게 밀집되며, baseline보다 안정적이고 정확함.
- **Variance Curve**: 샘플 수가 늘어남에 따라 분산이 빠르게 감소하며, 추정 신뢰도가 향상됨.

---

### 🔹 MVU Estimation: Large Sample (`x=2.0`, `y=3.0`, `n=500`)
- **Measurement Distribution**: 매우 조밀하게 중심에 집중된 분포를 보이며, 노이즈 효과가 최소화됨.
- **Estimate Distribution**: MVU 추정값이 거의 한 점에 모일 정도로 정확하게 수렴. 고신뢰도 추정이 가능함.
- **Variance Curve**: 분산이 거의 0에 수렴. Gaussian 환경에서 MVU의 일치성과 효율성이 명확히 드러남.

---

### 🔹 MLE Localization: Baseline (`x=5.0`, `y=6.0`, `σ=0.5`)
- **Measurement Circles**: 랜드마크가 고르게 배치되어 있으며, 측정 원들이 중심에서 균형 있게 교차함.
- **MLE Estimate**: 추정 위치는 실제 위치와 거의 일치하며, 안정적인 측정 조건에서 높은 정확도 확인.

---

### 🔹 MLE Localization: Corner Case (`x=1.0`, `y=1.0`, `σ=0.5`)
- **Measurement Circles**: 로봇이 한쪽 구석에 위치해 원이 한 방향으로 크게 퍼지며 교차가 불균형함.
- **MLE Estimate**: 실제 위치에 근접하지만 일부 왜곡된 추정 결과가 발생. 구조적 제약이 반영된 예시.

---

### 🔹 MLE Localization: Edge Case (`x=9.0`, `y=9.0`, `σ=0.5`)
- **Measurement Circles**: 좌측 하단으로 치우친 원들이 교차하는 형태로, 중심에서 크게 벗어난 배치 구조.
- **MLE Estimate**: 원의 교차 영역이 적어 추정의 불확실성이 높아짐. 추정은 실제 위치 주변에 위치함.

---

### 🔹 MLE Localization: Low Noise (`x=5.0`, `y=6.0`, `σ=0.1`)
- **Measurement Circles**: 반지름이 매우 작고, 교차 영역이 정밀하게 형성됨.
- **MLE Estimate**: 추정값은 실제 위치와 거의 완전히 일치. 노이즈가 작을 때 MLE의 정밀도가 극대화됨.

---

### 🔹 MLE Localization: High Noise (`x=5.0`, `y=6.0`, `σ=1.0`)
- **Measurement Circles**: 큰 반지름의 원들이 퍼지며, 교차 영역이 넓고 모호해짐.
- **MLE Estimate**: 실제 위치에서 조금 벗어난 지점을 추정함. 노이즈 크기가 증가할수록 추정 정확도는 떨어짐.
