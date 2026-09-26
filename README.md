# Visit with Us — Wellness Tourism Package Prediction

This repository contains the ML/MLOps implementation for predicting customer package purchase.

## Structure
- `tourism_project/data/` — dataset
- `tourism_project/model_building/` — validation and training scripts
- `tourism_project/deployment/` — Streamlit application, model artifact and deployment files
- `.github/workflows/pipeline.yml` — CI/CD pipeline

## Required GitHub Actions secrets
- `HF_TOKEN`
- `HF_DATASET_REPO`
- `HF_SPACE_REPO`

The Hugging Face Space should be public for the final submission.
