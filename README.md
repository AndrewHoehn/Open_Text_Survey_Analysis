# Survey Text Analysis Tool

A powerful Python tool for analyzing customer survey responses using AI-powered theme identification and automated translation. This tool processes open-text survey responses, identifies common themes using OpenAI's GPT models, and generates comprehensive reports with visualizations.

## Features

- **Automated Theme Detection**: Uses OpenAI GPT models to identify and categorize common themes in survey responses
- **Multi-language Support**: Automatically detects and translates non-English responses using DeepL or Google Translate
- **Smart Data Processing**: Handles incremental CSV imports, duplicate detection, and data validation
- **Comprehensive Reporting**: Generates markdown reports and CSV exports with theme analysis
- **Visualizations**: Creates charts showing theme distributions and satisfaction breakdowns
- **Flexible Analysis**: Segments satisfaction-related responses by satisfaction level while analyzing other questions holistically
- **Database Integration**: SQLite backend for efficient data management and incremental processing
- **Command-line Interface**: Full CLI support with options for batch processing and automation

## Quick Start

### Prerequisites

```bash
pip install pandas sqlite3 openai googletrans langdetect python-dotenv matplotlib deepl
```

### Required API Keys

- **OpenAI API Key**: For theme analysis (required)
- **DeepL API Key**: For translation (optional, will fall back to Google Translate)

### Basic Usage

1. **Set up environment variables**:
```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPL_API_KEY="your-deepl-api-key"  # Optional
```

2. **Run full analysis**:
```bash
python survey_analyzer.py --csv your_survey_data.csv --full
```

3. **Interactive mode** (discovers CSV files automatically):
```bash
python survey_analyzer.py
```

## Command Line Options

```bash
python survey_analyzer.py [OPTIONS]

Options:
  --csv PATH              Path to survey CSV file
  --db PATH              SQLite database file (default: survey.db)
  --full                 Run complete analysis pipeline
  --translate-only       Run only translation step
  --analyze-only         Run only theme analysis step
  --reports-only         Generate only charts and reports
  --assume-yes           Non-interactive mode (no user prompts)
  --db-action ACTION     Database action: 'replace' or 'append'
  --help                 Show this help message
```

## CSV File Requirements

Your CSV file should contain these columns:

### Required Columns
- **Overall satisfaction column**: Main satisfaction rating (e.g., "Very Satisfied", "Satisfied", etc.)
- **Contact Name**: Respondent name for quote attribution
- **Contact Email**: Email address (used for duplicate detection and employee filtering)
- **Contact Company**: Company name for quote attribution

### Open-Text Response Columns
The tool analyzes up to 5 open-text response columns. Default expected columns:
- "What is the main reason for your satisfaction rating?"
- "If [Company] could change one thing to make your job easier, what would it be?"
- "Is there anything you wish [Company] offered that we don't today?"
- "If you had 30 seconds with the [Company] executive team, what would you tell them?"
- "What changes in your industry are worrying you or keeping you up at night?"

### Column Name Customization
You can customize column names in the configuration:

```python
config = {
    'satisfaction_column': 'Your Satisfaction Column Name',
    'name_column': 'Your Name Column',
    'email_column': 'Your Email Column', 
    'company_column': 'Your Company Column',
}
```

## Key Features Explained

### Theme Analysis
- **AI-Powered**: Uses OpenAI GPT models to identify 3-8 distinct themes per question
- **Segmented Analysis**: Main satisfaction question is analyzed separately by satisfaction level
- **Representative Quotes**: Selects the most representative quote for each theme with full attribution
- **Exact Counts**: Assigns each response to a specific theme for precise counting

### Data Processing
- **Employee Filtering**: Automatically removes responses from company email domains
- **Duplicate Detection**: Uses email addresses and response fingerprints to avoid duplicate analysis
- **Incremental Updates**: Add new survey responses without re-analyzing existing data
- **Data Validation**: Checks for data quality issues and provides warnings

### Translation
- **Language Detection**: Automatically identifies non-English responses
- **Smart Translation**: Uses DeepL (premium) or Google Translate (free) with rate limiting
- **Preservation**: Keeps both original and translated text for reference

### Output Files

After analysis, you'll get:
- `survey_analysis.md` - Comprehensive markdown report
- `reports/` directory with PNG charts
- Individual CSV files for each question's themes
- SQLite database with all processed data

## Example Output

### Markdown Report Structure
```markdown
# Customer Survey Analysis Report

## Executive Summary
- Total Responses Analyzed: 1,041
- Analysis Date: January 15, 2024

## Satisfaction Distribution
| Satisfaction Level | Count |
|---|---|
| Satisfied | 451 |
| Very Satisfied | 328 |
| Neutral | 154 |

## Theme Analysis by Question

### What is the main reason for your satisfaction rating?

#### Satisfaction Level: Very Satisfied
- **Fast Response Time** — 45 responses
  > "The technical support team always responds quickly and knows exactly how to solve our problems"
  > *— John Smith, ABC Manufacturing*

- **Product Quality** — 38 responses  
  > "Equipment reliability has been outstanding - minimal downtime in 2 years"
  > *— Sarah Johnson, XYZ Industries*
```

### Chart Examples
- Stacked bar charts for satisfaction-segmented analysis
- Individual bar charts showing top themes per question
- High-resolution PNG files suitable for presentations

## Configuration Options

Create a configuration dictionary to customize behavior:

```python
config = {
    'ai_model': 'gpt-4o-mini',           # OpenAI model to use
    'ai_max_retries': 3,                 # API retry attempts
    'min_responses': 10,                 # Minimum responses required
    'large_dataset_threshold': 5000,     # Threshold for memory optimization
    'satisfaction_column': 'Satisfaction Rating',
    'name_column': 'Contact Name',
    'email_column': 'Contact Email',
    'company_column': 'Company Name'
}
```

## Advanced Usage

### Batch Processing
For automated processing of multiple survey files:

```bash
# Process multiple files with replace strategy
for file in *.csv; do
    python survey_analyzer.py --csv "$file" --db-action replace --assume-yes --full
done
```

### Incremental Analysis
To add new responses to existing analysis:

```bash
python survey_analyzer.py --csv new_responses.csv --db-action append --full
```

### Custom Analysis Pipeline
Run individual steps:

```bash
# 1. Import new data only
python survey_analyzer.py --csv data.csv --db-action append

# 2. Translate any untranslated responses  
python survey_analyzer.py --translate-only

# 3. Re-run theme analysis with updated data
python survey_analyzer.py --analyze-only

# 4. Generate fresh reports
python survey_analyzer.py --reports-only
```

## Supported AI Models

- **gpt-4o** - Most capable, higher cost
- **gpt-4o-mini** - Balanced performance and cost (recommended)
- **gpt-4-turbo** - Previous generation, reliable

## Performance Notes

- **Small datasets** (< 1,000 responses): 5-15 minutes
- **Medium datasets** (1,000-5,000 responses): 15-45 minutes  
- **Large datasets** (5,000+ responses): 45+ minutes

Processing time depends on:
- Number of non-English responses requiring translation
- API response times
- Dataset size and complexity

## Troubleshooting

### Common Issues

**"No themes identified"**
- Check OpenAI API key validity
- Ensure sufficient responses (minimum 10 per question)
- Verify internet connectivity

**"Translation failed"**  
- DeepL API key may be invalid or quota exceeded
- Google Translate rate limiting (add delays)
- Check internet connectivity

**"Missing required columns"**
- Verify CSV column names match expected names
- Use configuration to map custom column names
- Check for typos in column headers

**Memory issues with large datasets**
- Use `--assume-yes` to avoid interactive prompts
- Process in smaller batches
- Consider increasing system memory

### Debug Information

Run with verbose output to see detailed processing steps:
```bash
python survey_analyzer.py --csv data.csv --full --assume-yes
```

## Contributing

This tool is designed for customer survey analysis but can be adapted for other text analysis tasks. Key areas for contribution:

- Additional visualization types
- Support for more translation services
- Enhanced AI prompting strategies
- Performance optimizations for very large datasets
- Web interface development

## License

MIT License - see LICENSE file for details.

## Security Notes

- API keys are never stored in databases or files
- Temporary files are cleaned up automatically
- Employee email filtering prevents internal data inclusion
- All processing happens locally except for API calls
