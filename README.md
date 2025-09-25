# Survey Text Analysis Tool 

A Python tool that ingests a survey CSV, translates open-text answers, uses an LLM to generate **themes**, assigns every response to a theme for **exact counts**, and produces **charts** and a **markdown report**. Data is stored in **SQLite** so runs are repeatable and auditable.

> Works great for customer, employee, or product feedback surveys with open-ended questions.

---

## ✨ Features

- **CSV → SQLite import** with dedupe and basic data validation  
- **Translation to English** (DeepL preferred; Google Translate fallback)  
- **LLM-powered theming**
  - Proposes 4–8 concise themes per question
  - Classifies **every response** into a theme (stored in `theme_assignments`)
- **Representative quotes** with respondent name/company attribution
- **Charts** (PNG) and **Markdown report** summarizing themes and counts
- **CLI modes** to run the full pipeline or individual steps
- **Non-interactive mode** for automation/CI

---

## 📦 Requirements

- **Python** 3.10–3.12 recommended  
  (Python 3.13 may break `googletrans`; prefer DeepL or `deep-translator`)

### Install dependencies

Minimal set (required):
```bash
pip install pandas matplotlib python-dotenv
