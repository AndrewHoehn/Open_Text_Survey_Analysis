### Survey Text Analysis Tool

A Python tool that ingests a survey CSV, translates open-text answers, uses an LLM to generate themes, assigns every response to a theme for exact counts, and produces charts and a markdown report. Data is stored in SQLite so runs are repeatable and auditable.

Works great for customer, employee, or product feedback surveys with open-ended questions.

⸻

✨ Features
	•	CSV → SQLite import with dedupe and basic data validation
	•	Translation to English (DeepL preferred; Google Translate fallback)
	•	LLM-powered theming
	•	Proposes 4–8 concise themes per question
	•	Classifies every response into a theme (stored in theme_assignments)
	•	Representative quotes with respondent name/company attribution
	•	Charts (PNG) and Markdown report summarizing themes and counts
	•	CLI modes to run the full pipeline or individual steps
	•	Non-interactive mode for automation/CI

⸻

📦 Requirements
	•	Python 3.10–3.12 recommended
(Python 3.13 may break googletrans; prefer DeepL or deep-translator)

Install dependencies

Minimal set (required):

pip install pandas matplotlib python-dotenv

LLM (recommended):

pip install "openai>=1.45.0"

Translation (choose one):

# DeepL (recommended; requires API key)
pip install deepl

# or: googletrans (works best < Python 3.13)
pip install googletrans==4.0.0-rc1

Optional alternative translator (if on Python 3.13):

pip install deep-translator

(You’d need a small code tweak to use deep-translator instead of googletrans.)

Homebrew/Python note (macOS): If you hit externally-managed-environment errors (PEP 668), use a virtualenv, pyenv Python, or add --break-system-packages to pip. Best practice is a venv:

python3 -m venv .venv
source .venv/bin/activate


⸻

🔐 Environment Variables

Create a .env file (optional) in the project root:

OPENAI_API_KEY=sk-...
DEEPL_API_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxx
ASSUME_YES=1

	•	OPENAI_API_KEY — required for LLM theming
	•	DEEPL_API_KEY — optional; enables higher-quality translations
	•	ASSUME_YES — optional; non-interactive mode (skip prompts)

⸻

📁 Expected CSV Columns

Configure via config if your column names differ. Defaults:
	•	Satisfaction column: Overall, how satisfied are you with ...?
	•	Open text:
	•	What is the main reason for your satisfaction rating?
	•	If ... could change one thing to make your job easier, what would it be?
	•	Is there anything you wish ... offered that we don’t today?
	•	If you had 30 seconds with the executive team, what would you tell them?
	•	What changes in your industry are worrying you or keeping you up at night?
	•	Attribution:
	•	Contact Name
	•	Contact Account Name
	•	Contact Email

The first open-text question (“main reason”) is segmented by satisfaction; others are analyzed across all respondents.

⸻

🚀 Usage

Full pipeline

python survey_analyzer.py --csv "./survey.csv" --full --assume-yes

Step-by-step

# Import + translate only
python survey_analyzer.py --csv "./survey.csv" --translate-only

# Import + translate + analyze (no charts/reports)
python survey_analyzer.py --csv "./survey.csv" --analyze-only

# Charts + report from existing DB
python survey_analyzer.py --csv "./survey.csv" --reports-only

Options

--db <file>                SQLite DB path (default: cpm_survey.db)
--db-action replace|append Replace DB or append new rows on import
--assume-yes               Non-interactive mode (skip prompts)

During startup you can select an OpenAI model (default is a cost-efficient option). Models named gpt-5* may require the Responses API; the script uses Chat Completions and will still run with prior models (e.g., gpt-4o-mini).

⸻

🗄️ Data Model (SQLite)
	•	survey_responses — raw + translated text, satisfaction, contact fields
	•	themes — theme label/description per question (+segment) with one representative quote and attribution
	•	theme_assignments — one row per response→theme (source of truth for counts)
	•	analysis_runs — basic run metadata

Exact counts and charts are derived from theme_assignments (not LLM guesses).

⸻

📤 Outputs
	•	Charts (PNG): reports/
	•	Stacked bar: main reason by satisfaction
	•	Top themes: other questions
	•	Markdown report: cpm_survey_analysis.md
	•	Summary stats, per-question themes with counts
	•	One representative quote per theme (with name/company if available)

⸻

🔧 Configuration (advanced)

You can pass a config dict (or extend the script) to override:

config = {
  "ai_model": "gpt-4o-mini",
  "ai_max_retries": 3,
  "ai_base_delay": 0.1,
  "min_responses": 10,
  "large_dataset_threshold": 5000,
  # Column overrides:
  # "satisfaction_column": "...",
  # "name_column": "...",
  # "email_column": "...",
  # "company_column": "...",
}


⸻

🔒 Privacy & PII
	•	Representative quotes include the respondent name and company if available.
	•	If you need redaction (e.g., “First L., Company”), add a small post-processing step before writing reports.



🙌 Acknowledgments

Built to make open-text survey analysis fast, repeatable, and auditable — with exact counts, clear quotes, and minimal fuss.
