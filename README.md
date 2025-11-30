<!--
README for Employee Attrition Prediction & Analysis (DEPI)
Generated: 2025-11-30
-->

# 🚀 DEPI — Employee Attrition Prediction & Analysis

![DEPI](https://img.shields.io/badge/DEPI-Employee%20Attrition-blue?style=for-the-badge&logo=analytics)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![MLflow](https://img.shields.io/badge/MLflow-enabled-orange)
![Docker](https://img.shields.io/badge/Docker-ready-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

Short, production-minded project that predicts and analyzes employee attrition. DEPI brings together data preprocessing, model training (XGBoost), experiment tracking (MLflow), a REST API, and an interactive dashboard so HR teams and data practitioners can explore drivers of attrition and run individual predictions.

---

## ✨ Key Features

-   End-to-end ML pipeline: preprocessing → training → model artifacts (MLflow)
-   Trained XGBoost classification model for attrition prediction
-   REST API for real-time predictions
-   Interactive dashboard for exploratory analysis and visualizations
-   Docker + Docker Compose for reproducible local deployment
-   Example notebooks for preprocessing and model experiments

---

## 🧰 Technologies

-   Python 3.8+
-   Flask (API)
-   pandas, NumPy
-   scikit-learn, XGBoost
-   MLflow (experiment tracking & model registry)
-   Plotly / Matplotlib / Seaborn for visualizations
-   Docker & Docker Compose

---

## 📦 Repository layout

```
.
├── app/
│   ├── api.py            # REST API endpoints
│   ├── dashboard.py      # Dashboard/server code
│   ├── entry.py          # App entry point
│   ├── home.py
│   ├── model.py          # Model loading & predict helpers
│   ├── visual.py         # Visualization utilities
│   └── df_attr.csv       # Small dataset sample
├── data/                 # Raw, synthetic, and processed datasets
├── mlartifacts/          # Model artifacts (MLflow)
├── mlruns/               # MLflow runs
├── models/               # Saved model bundles
├── notebooks/            # Notebooks for preprocessing & experiments
├── backend.Dockerfile
├── frontend.Dockerfile
├── docker-compose.yml
├── requirements_backend.txt
├── requirements_frontend.txt
└── README.md
```

---

## ⚙️ Installation

Choose local Python environment or Docker. Docker is recommended for quick reproducible runs.

Prerequisites

-   Git
-   Python 3.8+
-   pip
-   Docker & Docker Compose (if using containers)

1. Clone the repository

```bash
# zsh
git clone https://github.com/MuhammedMahmoud0/Employee-Attrition-Prediction-and-Analysis-DEPI.git
cd Employee-Attrition-Prediction-and-Analysis-DEPI
```

2A) Local (venv) setup — backend only

```bash
# zsh
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_backend.txt
```

2B) Docker (recommended)

```bash
# zsh
docker compose build
docker compose up
```

Notes

-   Inspect `docker-compose.yml` for service ports and environment variables.
-   If you plan to run the frontend separately, install `requirements_frontend.txt`.

---

## ▶️ How to run

Local (development)

```bash
# zsh
source .venv/bin/activate
python app/entry.py
```

Docker (recommended)

```bash
# zsh
docker-compose up --build
```

After services start:

-   API: http://localhost:5000 (confirm in `docker-compose.yml`)
-   Dashboard: http://localhost:8501 (or the configured dashboard port)

Example API request

```bash
# zsh
curl -X POST "http://localhost:5000/predict" \
	-H "Content-Type: application/json" \
	-d '{"Age": 35, "MonthlyIncome": 5000, "JobRole": "Sales Executive", "OverTime": "Yes"}'
```

---

## 🖼️ Screenshots / Demo

<div align="center">
	<table>
		<tr>
			<td align="center" style="padding:8px">
				<figure>
					<img src="./assets/screenshot-dashboard1.png" alt="Dashboard 1" style="width:320px;max-width:45vw;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);" />
					<figcaption style="font-size:12px;margin-top:6px;color:#444">Dashboard — Overview</figcaption>
				</figure>
			</td>
			<td align="center" style="padding:8px">
				<figure>
					<img src="./assets/screenshot-dashboard2.png" alt="Dashboard 2" style="width:320px;max-width:45vw;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);" />
					<figcaption style="font-size:12px;margin-top:6px;color:#444">Dashboard — Filters & Charts</figcaption>
				</figure>
			</td>
		</tr>
		<tr>
			<td colspan="2" align="center" style="padding:8px">
				<figure>
					<img src="./assets/prediction.png" alt="Prediction response" style="width:560px;max-width:90vw;border-radius:8px;box-shadow:0 4px 12px rgba(0,0,0,0.08);" />
					<figcaption style="font-size:12px;margin-top:6px;color:#444">Sample prediction response</figcaption>
				</figure>
			</td>
		</tr>
	</table>
</div>

---

## 🧩 Data & Models

-   Preprocessed data: `data/Preprocessed_Data/`
-   Cleaned data: `data/*/Cleaned_Data/`
-   Trained models and artifacts: `mlartifacts/` and `mlruns/`

For retraining: review the notebooks in `notebooks/` and the training code in `app/`.

---

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Commit your changes with clear messages
4. Push and open a Pull Request

Contributing checklist

-   Follow PEP8 and code style in the repo
-   Add/adjust tests for new functionality
-   Keep changes focused and well-documented

Please open issues for bugs or feature requests and tag them appropriately.

---

## 👥 Collaborators

<div align="center">

<table>
	<tr>
		<td align="center" width="160">
			<img src="https://avatars.githubusercontent.com/u/179109266?v=4" width="96" alt="MuhammedMahmoud0" style="border-radius:50%"/>
			<p><strong>Muhammed Mahmmoud</strong><br/>
			<a href="https://github.com/MuhammedMahmoud0">@MuhammedMahmoud0</a><br/>
			<a href="mailto:muhammedmahmoud091@gmail.com">muhammedmahmoud091@gmail.com</a></p>
		</td>
		<td align="center" width="160">
			<img src="https://avatars.githubusercontent.com/u/167599516?v=4" width="96" alt="Abdallah1Atef" style="border-radius:50%"/>
			<p><strong>Abdallah Atef</strong><br/>
			<a href="https://github.com/Abdallah1Atef">@Abdallah1Atef</a></p>
		</td>
		<td align="center" width="160">
			<img src="https://avatars.githubusercontent.com/u/111690029?v=4" width="96" alt="Mohamed-ds-au" style="border-radius:50%"/>
			<p><strong>Mohamed</strong><br/>
			<a href="https://github.com/Mohamed-ds-au">@Mohamed-ds-au</a><br/>
		</td>
		<td align="center" width="160">
			<img src="https://avatars.githubusercontent.com/u/167798637?v=4" width="96" alt="ahmedemad" style="border-radius:50%"/>
			<p><strong>Ahmed Emad</strong><br/>
			<a href="https://github.com/Ahmed-Emad-Ds">@Ahmed-Emad-Ds </a><br/>
			<a href="mailto:ahmedemad7710@gmail.com">ahmedemad7710@gmail.com</a></p>
		</td>
	</tr>
</table>

</div>

## 🧪 Tests

Add tests under `tests/` and run with pytest:

```bash
# zsh
source .venv/bin/activate
pytest -q
```

Consider adding GitHub Actions for CI to run tests and linting automatically.

---

## ♻️ Roadmap

-   Add Postman collection and API docs
-   Add end-to-end integration tests and CI
-   Add model explainability pages (SHAP/LIME)
-   Deploy automated pipeline (CI/CD)

---

## 🙏 Acknowledgements

-   Datasets from Kaggle and synthetic generators
-   Project structure inspired by common ML deployment best practices

---
