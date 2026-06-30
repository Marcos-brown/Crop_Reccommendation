# Crop Recommendation Chatbot
An AI-powered agricultural advisory system that recommends optimal crops based on soil and environmental conditions, paired with a conversational agronomist assistant for personalized farming guidance.
## Overview

This project combines machine learning with conversational AI to help farmers and agricultural stakeholders make data-driven crop selection decisions. Users input soil and climate parameters and receive ranked crop recommendations, along with the ability to ask follow-up questions to an AI agronomist powered by the Anthropic Claude API.

## Features

- **Random Forest Classification** — Predicts the most suitable crop from 22 classes based on 7 soil and environmental features, achieving ~95.1% accuracy on the test set.
- **Hybrid Recommender** — Uses cosine similarity to surface alternative crop options beyond the top prediction, based on feature-space proximity.
- **AI Agronomist Chat** — A Claude API-powered conversational interface that answers follow-up questions about recommended crops, farming practices, and soil management.
- **Interactive Web App** — Built with Streamlit for an accessible, no-install user experience.
- **On-the-fly Training** — The model trains directly from the source CSV at runtime rather than relying on pre-saved model artifacts, keeping the pipeline transparent and reproducible.

## Tech Stack

| Component | Technology |
|---|---|
| Frontend / App | Streamlit |
| ML Model | scikit-learn (Random Forest) |
| Recommender | Cosine similarity (scikit-learn) |
| Conversational AI | Anthropic Claude API |
| Data | Kaggle crop recommendation dataset (3,200 records, 22 crop classes) |

## Dataset

The model is trained on a dataset of 3,200 records spanning 22 crop classes, using the following 7 features:

- Nitrogen (N)
- Phosphorus (P)
- Potassium (K)
- Temperature
- Humidity
- Soil pH
- Rainfall

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/<your-username>/<your-repo-name>.git
   cd <your-repo-name>
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY="your-api-key-here"  # On Windows: set ANTHROPIC_API_KEY=your-api-key-here
   ```
   Or create a `.env` file in the project root:
   ```
   ANTHROPIC_API_KEY=your-api-key-here
   ```

## Usage

Run the Streamlit app from the project root directory:

```bash
streamlit run app.py
```

Then open the local URL shown in your terminal (typically `http://localhost:8501`).

1. Enter your soil and environmental parameters (N, P, K, temperature, humidity, pH, rainfall).
2. View the top crop recommendation along with similar alternatives.
3. Chat with the AI agronomist for advice on planting, soil prep, or care for the recommended crop.

## Project Structure

```
.
├── app.py                 # Main Streamlit application
├── data/
│   └── crop_data.csv      # Source dataset
├── model/
│   └── train_model.py     # Model training logic (runs on app startup)
├── requirements.txt        # Python dependencies
└── README.md
```

## Model Performance

The Random Forest classifier achieves approximately **95.1% accuracy** on the held-out test set across all 22 crop classes.

## Roadmap

- [ ] Add support for region-specific crop datasets
- [ ] Incorporate weather API integration for live climate data
- [ ] Expand recommender to factor in market price trends
- [ ] Add multilingual support for the chat interface

## Contributing

Contributions, issues, and feature requests are welcome. Feel free to check the issues page or open a pull request.

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset sourced from Kaggle's crop recommendation dataset
- Conversational AI powered by the [Anthropic Claude API](https://docs.claude.com)
