#!/usr/bin/env python3

import os
import re
import json
import time
import sqlite3
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import pandas as pd

# Optional deps
try:
    from dotenv import load_dotenv

    load_dotenv()
    ENV_LOADED = True
except Exception:
    ENV_LOADED = False

# Translation libs (DeepL preferred, Google Translate fallback)
try:
    import deepl

    DEEPL_AVAILABLE = True
except ImportError as e:
    print(f"DeepL not available: {e}")
    DEEPL_AVAILABLE = False
except Exception as e:
    print(f"DeepL import error: {e}")
    DEEPL_AVAILABLE = False

try:
    from googletrans import Translator

    # Test if it actually works
    test_translator = Translator()
    GOOGLETRANS_AVAILABLE = True
except ImportError as e:
    print(f"Google Translate not available: {e}")
    print("💡 To install: pip install googletrans==4.0.0rc1")
    GOOGLETRANS_AVAILABLE = False
except Exception as e:
    print(f"Google Translate import error: {e}")
    print("💡 Try: pip install googletrans==4.0.0rc1")
    GOOGLETRANS_AVAILABLE = False

# OpenAI SDK (modern client)
OPENAI_OK = False
try:
    from openai import OpenAI

    OPENAI_OK = True
    print("✅ OpenAI library imported successfully")
except ImportError as e:
    print(f"❌ OpenAI library not installed: {e}")
    print("💡 To install: pip install openai")
    OPENAI_OK = False
except Exception as e:
    print(f"❌ OpenAI library import failed: {e}")
    OPENAI_OK = False

# Plotting (matplotlib only, no seaborn)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Language detection
try:
    from langdetect import detect

    LANGDETECT_AVAILABLE = True
except ImportError as e:
    print(f"Language detection not available: {e}")
    LANGDETECT_AVAILABLE = False
except Exception as e:
    print(f"Language detection import error: {e}")
    LANGDETECT_AVAILABLE = False


def slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "-", s.strip())
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s[:max_len]


class CPMSurveyAnalyzer:
    def __init__(
        self,
        csv_file: str,
        db_file: str = "cpm_survey.db",
        assume_yes: bool = False,
        config: Dict = None,
    ):
        self.csv_file = csv_file
        self.db_file = db_file
        self.assume_yes = assume_yes
        self.config = config or {}
        self.conn: Optional[sqlite3.Connection] = None

        # Columns (ensure these exactly match your CSV—use the resolved names we detected)
        self.SATISFACTION_COLUMN = self.config.get(
            "satisfaction_column", "Overall, how satisfied are you with CPM?"
        )
        self.OPEN_TEXT_COLUMNS = [
            "What is the main reason for your satisfaction rating?",
            "If CPM could change one thing to make your job easier, what would it be?",
            "Is there anything you wish CPM offered that we don’t today?",
            "If you had 30 seconds with the CPM executive team, what would you tell them?",
            "What changes in your industry are worrying you or keeping you up at night?",
        ]
        self.NAME_COLUMN = self.config.get("name_column", "Contact Name")
        self.COMPANY_COLUMN = self.config.get("company_column", "Contact Account Name")
        self.EMAIL_COLUMN = self.config.get("email_column", "Contact Email")

        # API / translation setup (order matters: keys first, then translator)
        self.openai_key = None
        self.deepl_key = None
        self.translation_service = None
        self.translator = None

        self._setup_api_keys()
        self._setup_translator()

        print("🚀 CPM Survey Analyzer initialized!")
        print(f"📊 CSV file: {csv_file}")
        print(f"🗄️  Database: {db_file}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.conn:
            self.conn.close()
            print("🔐 Database connection closed")

    def _setup_api_keys(self) -> bool:
        print("\n🔑 Checking API keys...")
        print(f"🔧 Debug - OPENAI_OK at startup: {OPENAI_OK}")

        self.openai_key = os.getenv("OPENAI_API_KEY") or ""
        if self.openai_key:
            print("✅ OPENAI_API_KEY found in environment")
            print(f"🔧 Debug - API key length: {len(self.openai_key)} chars")
        else:
            print("⚠️  OPENAI_API_KEY not found")
            if not self.assume_yes:
                try:
                    self.openai_key = input(
                        "Enter OpenAI API key (or press Enter to continue without AI): "
                    ).strip()
                    if self.openai_key:
                        print(
                            f"🔧 Debug - Entered API key length: {len(self.openai_key)} chars"
                        )
                except EOFError:
                    self.openai_key = ""
            if not self.openai_key:
                print(
                    "ℹ️  Proceeding without AI (theme naming/assignment will be skipped)"
                )

        self.deepl_key = os.getenv("DEEPL_API_KEY") or ""
        if not self.deepl_key and DEEPL_AVAILABLE:
            print("⚠️  DEEPL_API_KEY not found")
            if not self.assume_yes:
                try:
                    self.deepl_key = input(
                        "Enter DeepL API key (or press Enter to use Google Translate): "
                    ).strip()
                except EOFError:
                    self.deepl_key = ""

        if self.deepl_key and DEEPL_AVAILABLE:
            print("✅ DEEPL_API_KEY found; will use DeepL for translation")
        elif GOOGLETRANS_AVAILABLE:
            print("✅ Using Google Translate (free) for translation")
        else:
            print("⚠️  No translation service available; text will not be translated")
        return True

    def _setup_translator(self):
        print("\n🌐 Setting up translation service...")
        if self.deepl_key and DEEPL_AVAILABLE:
            try:
                self.translator = deepl.Translator(self.deepl_key)
                self.translation_service = "deepl"
                print("✅ DeepL translator initialized")
                return
            except Exception as e:
                print(f"⚠️  DeepL init failed: {e}")

        if GOOGLETRANS_AVAILABLE:
            try:
                self.translator = Translator()
                self.translation_service = "google"
                print("✅ Google Translate initialized")
                return
            except Exception as e:
                print(f"⚠️  Google Translate init failed: {e}")

        self.translator = None
        self.translation_service = None
        print("❌ No translator available")

    def setup_database(self):
        print("\n🗄️  Setting up database...")
        self.conn = sqlite3.connect(self.db_file)
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS survey_responses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                satisfaction_rating TEXT,
                main_reason TEXT,
                main_reason_translated TEXT,
                main_reason_language TEXT,
                job_easier TEXT,
                job_easier_translated TEXT,
                job_easier_language TEXT,
                wish_offered TEXT,
                wish_offered_translated TEXT,
                wish_offered_language TEXT,
                exec_message TEXT,
                exec_message_translated TEXT,
                exec_message_language TEXT,
                industry_worries TEXT,
                industry_worries_translated TEXT,
                industry_worries_language TEXT,
                contact_name TEXT,
                contact_company TEXT,
                contact_email TEXT,
                is_cpm_employee BOOLEAN DEFAULT 0,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS themes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                question_column TEXT,
                theme_name TEXT,
                theme_description TEXT,
                satisfaction_segment TEXT,
                representative_quote TEXT,
                representative_respondent TEXT,
                representative_company TEXT,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS theme_assignments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                response_id INTEGER,
                question_column TEXT,
                theme_name TEXT,
                satisfaction_segment TEXT,
                confidence REAL,
                created_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS analysis_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_responses INTEGER,
                new_responses INTEGER,
                translated_responses INTEGER,
                themes_identified INTEGER,
                status TEXT
            )
        """)
        self.conn.commit()
        print("✅ Database tables created/verified")

    def _safe_str(self, value) -> str:
        """Safely convert pandas values to string, handling NaN"""
        if pd.isna(value):
            return ""
        return str(value).strip()

    def _is_english(self, text: str) -> bool:
        """Detect if text is in English using simple heuristics and langdetect"""
        if not text or len(text.strip()) < 3:
            return True  # Assume very short text is English

        text = text.strip()

        # Simple heuristic: if text is mostly ASCII and contains common English words
        ascii_ratio = sum(1 for c in text if ord(c) < 128) / len(text)
        if ascii_ratio > 0.9:
            common_english = [
                "the",
                "and",
                "is",
                "to",
                "of",
                "a",
                "in",
                "that",
                "it",
                "with",
                "for",
                "as",
                "was",
                "on",
                "are",
                "you",
                "this",
                "be",
                "at",
                "have",
            ]
            text_lower = text.lower()
            english_word_count = sum(1 for word in common_english if word in text_lower)
            if (
                english_word_count >= 2 or len(text) < 20
            ):  # Short text or has common English words
                return True

        # Use langdetect if available for more accurate detection
        if LANGDETECT_AVAILABLE:
            try:
                detected_lang = detect(text)
                return detected_lang == "en"
            except:
                # If detection fails, fall back to heuristic result
                pass

        # Default: if mostly ASCII, probably English
        return ascii_ratio > 0.9

    def _validate_csv_data(self, df: pd.DataFrame) -> Tuple[bool, List[str]]:
        """Validate CSV data quality and return validation status with issues"""
        issues = []

        # Check minimum response threshold
        min_responses = self.config.get("min_responses", 10)
        if len(df) < min_responses:
            issues.append(
                f"Only {len(df)} responses found, minimum {min_responses} required"
            )

        # Validate email formats
        if self.EMAIL_COLUMN in df.columns:
            valid_emails = (
                df[self.EMAIL_COLUMN]
                .astype(str)
                .str.contains(r"^[^@]+@[^@]+\.[^@]+$", na=False)
            )
            invalid_count = len(df) - valid_emails.sum()
            if invalid_count > len(df) * 0.5:  # More than 50% invalid emails
                issues.append(f"{invalid_count} invalid email formats detected")

        # Check for suspicious duplicate text patterns
        for col in self.OPEN_TEXT_COLUMNS:
            if col in df.columns:
                non_empty = df[col].dropna().astype(str).str.strip()
                non_empty = non_empty[non_empty != ""]
                if len(non_empty) > 0:
                    duplicates = non_empty.duplicated().sum()
                    if duplicates > len(non_empty) * 0.8:  # More than 80% duplicates
                        issues.append(
                            f"High duplicate rate ({duplicates}/{len(non_empty)}) in '{col}'"
                        )

        # Check for extremely short responses that might indicate data quality issues
        for col in self.OPEN_TEXT_COLUMNS:
            if col in df.columns:
                short_responses = df[col].astype(str).str.len() < 3
                short_count = short_responses.sum()
                if short_count > len(df) * 0.9:  # More than 90% are very short
                    issues.append(
                        f"Most responses in '{col}' are extremely short (< 3 chars)"
                    )

        return len(issues) == 0, issues

    def import_csv_data(self) -> bool:
        print("\n📥 Importing CSV data...")
        try:
            df = pd.read_csv(self.csv_file)
            print(f"📊 Loaded {len(df):,} rows from CSV")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return False

        # Check existing database status
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM survey_responses")
        existing_count = cur.fetchone()[0]

        if existing_count > 0:
            print(f"🗄️  Database already contains {existing_count:,} responses")

            # Check for command line db-action argument
            db_action = getattr(self, "_db_action", None)

            if db_action == "replace":
                print("🗑️  Clearing existing database (--db-action=replace)...")
                self.clear_database()
                print("✅ Database cleared")
            elif db_action == "append":
                print("➕ Will append new responses only (--db-action=append)")
            elif not self.assume_yes:
                print("\nDatabase Management Options:")
                print("  1. Replace all data (clear database and import fresh)")
                print("  2. Append new responses only (skip duplicates)")
                print("  3. Cancel import")

                while True:
                    try:
                        choice = input("\nSelect option (1-3): ").strip()
                        if choice == "1":
                            print("🗑️  Clearing existing database...")
                            self.clear_database()
                            print("✅ Database cleared")
                            break
                        elif choice == "2":
                            print("➕ Will append new responses only")
                            break
                        elif choice == "3":
                            print("❌ Import cancelled")
                            return False
                        else:
                            print("❌ Please enter 1, 2, or 3")
                    except (EOFError, KeyboardInterrupt):
                        print("\n❌ Import cancelled")
                        return False
            else:
                print("⚠️  Non-interactive mode: will append new responses only")
        else:
            print("📝 Database is empty - will import all data")

        required = [
            self.SATISFACTION_COLUMN,
            self.EMAIL_COLUMN,
            self.NAME_COLUMN,
            self.COMPANY_COLUMN,
        ] + self.OPEN_TEXT_COLUMNS
        missing = [c for c in required if c not in df.columns]
        if missing:
            print(f"❌ Missing required columns: {missing}")
            return False

        # Validate data quality
        is_valid, validation_issues = self._validate_csv_data(df)
        if validation_issues:
            print("⚠️  Data validation warnings:")
            for issue in validation_issues:
                print(f"   • {issue}")
            if not is_valid and not self.assume_yes:
                try:
                    response = (
                        input("Continue despite validation issues? (y/N): ")
                        .strip()
                        .lower()
                    )
                    if response not in ["y", "yes"]:
                        print("❌ Import cancelled due to validation issues")
                        return False
                except EOFError:
                    print("❌ Import cancelled due to validation issues")
                    return False

        start = len(df)
        df = df[~df[self.EMAIL_COLUMN].astype(str).str.lower().str.endswith("cpm.net")]
        print(f"🚫 Filtered out {start - len(df)} CPM employee rows")

        cur = self.conn.cursor()
        cur.execute(
            "SELECT LOWER(TRIM(contact_email)) FROM survey_responses WHERE contact_email IS NOT NULL AND TRIM(contact_email) != ''"
        )
        existing_emails = set([row[0] for row in cur.fetchall() if row[0]])

        print(f"🔍 Found {len(existing_emails):,} existing email addresses in database")

        new_count = 0
        skipped_count = 0
        duplicate_emails = 0
        duplicate_manual = 0

        for _, row in df.iterrows():
            email = self._safe_str(row.get(self.EMAIL_COLUMN)).lower().strip()

            # Check email-based deduplication first
            if email:
                if email in existing_emails:
                    duplicate_emails += 1
                    continue
            else:
                # For rows without email, use name+company+satisfaction as key
                key = (
                    self._safe_str(row.get(self.NAME_COLUMN)) or "",
                    self._safe_str(row.get(self.COMPANY_COLUMN)) or "",
                    self._safe_str(row.get(self.SATISFACTION_COLUMN)) or "",
                )

                # Skip if all key fields are empty
                if not any(key):
                    skipped_count += 1
                    continue

                cur.execute(
                    "SELECT 1 FROM survey_responses WHERE COALESCE(contact_name,'')=? AND COALESCE(contact_company,'')=? AND COALESCE(satisfaction_rating,'')=?",
                    key,
                )
                exists = cur.fetchone() is not None
                if exists:
                    duplicate_manual += 1
                    continue

            cur.execute(
                """
                INSERT INTO survey_responses (
                    satisfaction_rating, main_reason, job_easier, wish_offered,
                    exec_message, industry_worries, contact_name, contact_company,
                    contact_email, is_cpm_employee
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
                (
                    self._safe_str(row.get(self.SATISFACTION_COLUMN)) or None,
                    self._safe_str(row.get(self.OPEN_TEXT_COLUMNS[0])) or None,
                    self._safe_str(row.get(self.OPEN_TEXT_COLUMNS[1])) or None,
                    self._safe_str(row.get(self.OPEN_TEXT_COLUMNS[2])) or None,
                    self._safe_str(row.get(self.OPEN_TEXT_COLUMNS[3])) or None,
                    self._safe_str(row.get(self.OPEN_TEXT_COLUMNS[4])) or None,
                    self._safe_str(row.get(self.NAME_COLUMN)) or None,
                    self._safe_str(row.get(self.COMPANY_COLUMN)) or None,
                    email or None,
                ),
            )
            new_count += 1
            if email:
                existing_emails.add(email)

        self.conn.commit()

        # Provide detailed import summary
        cur.execute("SELECT COUNT(*) FROM survey_responses")
        final_count = cur.fetchone()[0]

        total_processed = len(df)
        total_duplicates = duplicate_emails + duplicate_manual

        print(f"\n📊 Import Summary:")
        print(f"   • CSV rows processed: {total_processed:,}")
        print(f"   • New responses imported: {new_count:,}")
        print(f"   • Duplicates skipped (email): {duplicate_emails:,}")
        print(f"   • Duplicates skipped (manual key): {duplicate_manual:,}")
        if skipped_count > 0:
            print(f"   • Rows skipped (insufficient data): {skipped_count:,}")
        print(f"   • Database total: {final_count:,} responses")

        if new_count == 0 and total_duplicates > 0:
            print("ℹ️  All CSV data already exists in database - no new responses added")
        elif new_count == 0:
            print("⚠️  No responses were imported - check your CSV data")
        return True

    def detect_and_translate_responses(self):
        if not self.translator:
            print("⚠️  No translator available; skipping translation")
            return

        start_time = time.time()
        print(f"\n🌍 Starting translation using {self.translation_service.upper()}...")

        cur = self.conn.cursor()
        text_cols = [
            (
                "main_reason",
                "main_reason_translated",
                "main_reason_language",
                "Main Reason",
            ),
            (
                "job_easier",
                "job_easier_translated",
                "job_easier_language",
                "Job Easier",
            ),
            (
                "wish_offered",
                "wish_offered_translated",
                "wish_offered_language",
                "Wish Offered",
            ),
            (
                "exec_message",
                "exec_message_translated",
                "exec_message_language",
                "Executive Message",
            ),
            (
                "industry_worries",
                "industry_worries_translated",
                "industry_worries_language",
                "Industry Worries",
            ),
        ]

        # First, count total items to translate
        total_to_translate = 0
        for orig, trans, lang, name in text_cols:
            cur.execute(
                f"SELECT COUNT(*) FROM survey_responses WHERE {orig} IS NOT NULL AND TRIM({orig}) != '' AND {trans} IS NULL"
            )
            count = cur.fetchone()[0]
            total_to_translate += count

        if total_to_translate == 0:
            print("✅ All texts already translated, skipping translation step")
            return

        print(f"📝 Found {total_to_translate:,} text entries to translate")

        translated_count = 0
        english_skipped = 0

        for orig, trans, lang, name in text_cols:
            cur.execute(
                f"SELECT id, {orig} FROM survey_responses WHERE {orig} IS NOT NULL AND TRIM({orig}) != '' AND {trans} IS NULL"
            )
            rows = cur.fetchall()

            if not rows:
                continue

            print(f"\n🔄 Processing {len(rows):,} responses for '{name}'...")

            # First pass: detect languages and count what needs translation
            english_count = 0
            needs_translation = []

            for r_id, text in rows:
                text_str = str(text).strip()
                if self._is_english(text_str):
                    # Mark as English, no translation needed
                    cur.execute(
                        f"UPDATE survey_responses SET {trans}=?, {lang}=? WHERE id=?",
                        (text_str, "en", r_id),
                    )
                    english_count += 1
                    english_skipped += 1
                else:
                    needs_translation.append((r_id, text_str))

            if english_count > 0:
                print(
                    f"   ✅ Marked {english_count:,} responses as English (no translation needed)"
                )

            if not needs_translation:
                print(f"   ℹ️  All responses for '{name}' are in English")
                continue

            print(
                f"   🌍 Translating {len(needs_translation):,} non-English responses..."
            )

            for i, (r_id, text_str) in enumerate(needs_translation, 1):
                # Show progress every 5 items or on last item
                if i % 5 == 0 or i == len(needs_translation):
                    progress = i / len(needs_translation) * 100
                    elapsed = time.time() - start_time
                    rate = translated_count / elapsed if elapsed > 0 else 0
                    remaining_translations = (
                        total_to_translate - translated_count - english_skipped
                    )
                    eta = remaining_translations / rate if rate > 0 else 0
                    print(
                        f"   📊 {i}/{len(needs_translation)} ({progress:.1f}%) | Rate: {rate:.1f}/s | ETA: {eta / 60:.1f}m",
                        end="\r",
                    )

                try:
                    if self.translation_service == "deepl":
                        res = self.translator.translate_text(
                            text_str, target_lang="EN-US"
                        )
                        translated_text = res.text
                        source_lang = getattr(res, "detected_source_lang", "auto")
                    elif self.translation_service == "google":
                        res = self.translator.translate(text_str, dest="en")
                        translated_text = res.text
                        source_lang = getattr(res, "src", "auto")
                    else:
                        translated_text = text_str
                        source_lang = "unknown"
                    cur.execute(
                        f"UPDATE survey_responses SET {trans}=?, {lang}=? WHERE id=?",
                        (translated_text, source_lang, r_id),
                    )
                    translated_count += 1
                except Exception as e:
                    print(f"\n⚠️  Translation error for ID {r_id}: {e}")
                    cur.execute(
                        f"UPDATE survey_responses SET {trans}=?, {lang}=? WHERE id=?",
                        (text_str, "error", r_id),
                    )

                # Rate limiting for Google Translate
                if self.translation_service == "google":
                    time.sleep(0.05)

            print()  # New line after progress indicator

            # Commit after each column to save progress
            self.conn.commit()
            print(f"   ✅ Completed '{name}' translations")

        total_time = time.time() - start_time
        total_processed = translated_count + english_skipped

        print(f"\n🎉 Translation complete! Summary:")
        print(f"   • Total texts processed: {total_processed:,}")
        print(f"   • English responses (no translation): {english_skipped:,}")
        print(f"   • Non-English responses translated: {translated_count:,}")
        print(
            f"   • Time saved by skipping English: ~{(english_skipped * 0.2):.1f} minutes"
        )
        print(f"   • Total processing time: {total_time / 60:.1f} minutes")

    def _openai_client(self) -> Optional["OpenAI"]:
        print(
            f"      🔧 Debug - OPENAI_OK: {OPENAI_OK}, API key exists: {bool(self.openai_key)}"
        )
        if not OPENAI_OK:
            print(f"      ❌ OpenAI library not available")
            return None
        if not self.openai_key:
            print(f"      ❌ No OpenAI API key provided")
            return None
        try:
            client = OpenAI(api_key=self.openai_key)
            print(f"      ✅ OpenAI client created successfully")

            # Test the client with a simple API call
            try:
                test_response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": "Say 'test'"}],
                    max_tokens=5,
                )
                print(f"      ✅ OpenAI API connection verified")
            except Exception as e:
                print(f"      ⚠️  OpenAI API test failed: {e}")
                print(f"      ℹ️  Client created but API may not be working")

            return client
        except Exception as e:
            print(f"      ❌ OpenAI client init failed: {e}")
            return None

    def _db_col_for_question(self, question_column: str) -> Optional[str]:
        mapping = {
            self.OPEN_TEXT_COLUMNS[0]: "main_reason_translated",
            self.OPEN_TEXT_COLUMNS[1]: "job_easier_translated",
            self.OPEN_TEXT_COLUMNS[2]: "wish_offered_translated",
            self.OPEN_TEXT_COLUMNS[3]: "exec_message_translated",
            self.OPEN_TEXT_COLUMNS[4]: "industry_worries_translated",
        }
        return mapping.get(question_column)

    def _fetch_responses(
        self, db_col: str, satisfaction_segment: Optional[str]
    ) -> List[Tuple[int, str, str, str, str]]:
        cur = self.conn.cursor()
        base = f"SELECT id, {db_col}, contact_name, contact_company, satisfaction_rating FROM survey_responses WHERE {db_col} IS NOT NULL AND TRIM({db_col}) != ''"
        params: List = []
        if satisfaction_segment:
            base += " AND satisfaction_rating = ?"
            params.append(satisfaction_segment)
        cur.execute(base, params)
        out = []
        for r in cur.fetchall():
            out.append((r[0], str(r[1]).strip(), r[2] or "", r[3] or "", r[4] or ""))
        return out

    def _fetch_responses_chunked(
        self, db_col: str, satisfaction_segment: Optional[str], chunk_size: int = 1000
    ):
        """Memory-efficient generator for processing large datasets"""
        cur = self.conn.cursor()
        base = f"SELECT id, {db_col}, contact_name, contact_company, satisfaction_rating FROM survey_responses WHERE {db_col} IS NOT NULL AND TRIM({db_col}) != ''"
        params: List = []
        if satisfaction_segment:
            base += " AND satisfaction_rating = ?"
            params.append(satisfaction_segment)

        # Add LIMIT and OFFSET for chunking
        base += f" LIMIT {chunk_size} OFFSET ?"

        offset = 0
        while True:
            chunk_params = params + [offset]
            cur.execute(base, chunk_params)
            rows = cur.fetchall()
            if not rows:
                break

            chunk = []
            for r in rows:
                chunk.append(
                    (r[0], str(r[1]).strip(), r[2] or "", r[3] or "", r[4] or "")
                )

            yield chunk
            offset += chunk_size

    def _rate_limited_ai_call(self, func, *args, **kwargs):
        """Execute AI API call with rate limiting and retry logic"""
        max_retries = self.config.get("ai_max_retries", 3)
        base_delay = self.config.get("ai_base_delay", 0.5)  # Increased from 0.1 to 0.5

        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    delay = base_delay * (2**attempt)  # Exponential backoff
                    time.sleep(delay)
                else:
                    time.sleep(base_delay)  # Basic rate limiting

                # Add timeout to API calls
                if "timeout" not in kwargs:
                    kwargs["timeout"] = self.config.get(
                        "ai_timeout", 300
                    )  # 5 minute timeout for large batch processing

                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                print(
                    f" ⚠️  Retry {attempt + 1}/{max_retries} ({e})", end="", flush=True
                )

    def _ai_propose_themes(
        self,
        client: "OpenAI",
        question_column: str,
        responses: List[str],
        satisfaction_segment: Optional[str],
    ) -> List[Dict]:
        # Filter to only English responses and take a larger sample for better theme detection
        english_responses = [
            r for r in responses if self._is_english(r) and len(r.strip()) > 10
        ]
        preview = english_responses[:200]  # Increased sample size

        if len(preview) < 3:
            return []  # Not enough data for meaningful themes

        prompt = f"""
You are analyzing customer survey responses for CPM, an industrial equipment manufacturing company.
Identify the most common distinct themes that emerge from these customer responses.

Question: "{question_column}"
{("Customer satisfaction level: " + satisfaction_segment) if satisfaction_segment else "All satisfaction levels"}
Total responses analyzed: {len(preview)}

Requirements:
1. Create 4-8 specific, actionable themes (avoid generic categories)
2. Theme names should be 2-5 words, business-focused
3. Each theme should represent at least 5% of responses
4. Pick the clearest, most representative English quote for each theme
5. Focus on themes that would be useful for business decision-making

Sample responses:
{chr(10).join([f'{i + 1}. "{t}"' for i, t in enumerate(preview)])}

Examples of good theme names:
- "Product Quality Issues"
- "Customer Support Speed"
- "Pricing Concerns"
- "Technical Training Needs"
- "Equipment Reliability"

Return strict JSON format:
{{
  "themes": [{{"name": "Theme Name","description": "Brief description","representative_quote": "Exact quote from sample"}}]
}}
"""
        try:
            ai_model = self.config.get("ai_model", "gpt-4o-mini")
            # Prepare API call parameters - GPT-5 models don't support temperature
            api_params = {
                "model": ai_model,
                "messages": [{"role": "user", "content": prompt}],
            }
            if not ai_model.startswith("gpt-5"):
                api_params["temperature"] = 0.2

            r = self._rate_limited_ai_call(client.chat.completions.create, **api_params)

            # Debug the response
            if not r or not r.choices:
                print("❌ Empty response from OpenAI API")
                return []

            content = r.choices[0].message.content
            if not content or not content.strip():
                print("❌ Empty content from OpenAI API response")
                return []

            # Clean and try to parse JSON
            content = content.strip()

            # Remove markdown code blocks if present
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

            # Debug output for troubleshooting
            if len(content) < 50:
                print(f"⚠️  Short API response: '{content}'")

            data = json.loads(content)
            themes = data.get("themes", [])
            clean = []
            for th in themes:
                nm = th.get("name", "").strip()[:120]
                ds = th.get("description", "").strip()[:500]
                qt = th.get("representative_quote", "").strip()
                if nm:
                    clean.append(
                        {"name": nm, "description": ds, "representative_quote": qt}
                    )
            return clean
        except json.JSONDecodeError as e:
            print(f"❌ JSON parsing error: {e}")
            print(f"❌ Raw API response content: '{content[:200]}...' (truncated)")
            return []
        except Exception as e:
            print(f"❌ AI theme proposal error: {e}")
            return []

    def _ai_assign_themes(
        self,
        client: "OpenAI",
        question_column: str,
        responses: List[str],
        theme_names: List[str],
    ) -> List[int]:
        assignments: List[int] = []
        if not theme_names:
            return [0] * len(responses)

        BATCH = 20  # Reduced batch size to avoid timeouts
        total_batches = (len(responses) + BATCH - 1) // BATCH
        start_time = time.time()

        print(
            f"      📦 Processing {len(responses)} responses in {total_batches} batches of {BATCH}..."
        )

        for batch_num, i in enumerate(range(0, len(responses), BATCH), 1):
            batch = responses[i : i + BATCH]

            # Show progress
            elapsed = time.time() - start_time
            if batch_num > 1:
                avg_time_per_batch = elapsed / (batch_num - 1)
                remaining_batches = total_batches - batch_num
                eta_seconds = remaining_batches * avg_time_per_batch
                eta_minutes = eta_seconds / 60
                print(
                    f"      📊 Batch {batch_num}/{total_batches} ({len(batch)} responses) | ETA: {eta_minutes:.1f}m",
                    end="",
                    flush=True,
                )
            else:
                print(
                    f"      📊 Batch {batch_num}/{total_batches} ({len(batch)} responses)",
                    end="",
                    flush=True,
                )

            prompt = f"""
Classify each response into exactly one of the following themes by index.

Themes:
{chr(10).join([f"{k}. {nm}" for k, nm in enumerate(theme_names)])}

Return a JSON list of integers (theme indices), one per response in the same order. No other text.

Responses:
{chr(10).join([f"{j + 1}. {t}" for j, t in enumerate(batch)])}
"""
            # Try up to 2 times for this batch
            batch_success = False
            for batch_attempt in range(2):
                try:
                    ai_model = self.config.get("ai_model", "gpt-4o-mini")
                    if batch_attempt == 0:
                        print(" 🔄 Calling API...", end="", flush=True)
                    else:
                        print(f" 🔁 Retry {batch_attempt}...", end="", flush=True)

                    api_start = time.time()
                    # Prepare API call parameters - GPT-5 models don't support temperature
                    api_params = {
                        "model": ai_model,
                        "messages": [{"role": "user", "content": prompt}],
                    }
                    if not ai_model.startswith("gpt-5"):
                        api_params["temperature"] = 0.0 if batch_attempt == 0 else 0.1

                    r = self._rate_limited_ai_call(
                        client.chat.completions.create, **api_params
                    )
                    api_time = time.time() - api_start
                    print(f" ({api_time:.1f}s)", end="", flush=True)
                    content = r.choices[0].message.content.strip()

                    # Clean JSON response
                    if content.startswith("```json"):
                        content = content[7:]
                    if content.startswith("```"):
                        content = content[3:]
                    if content.endswith("```"):
                        content = content[:-3]
                    content = content.strip()

                    idxs = json.loads(content)
                    if not isinstance(idxs, list) or len(idxs) != len(batch):
                        if batch_attempt == 0:
                            print(
                                f" ⚠️  Wrong count ({len(idxs) if isinstance(idxs, list) else 'invalid'} vs {len(batch)})",
                                end="",
                                flush=True,
                            )
                            continue  # Try again
                        else:
                            raise ValueError(
                                f"Bad classifier output shape: expected {len(batch)}, got {len(idxs) if isinstance(idxs, list) else type(idxs)}"
                            )

                    # Validate and clamp theme indices to valid range
                    valid_idxs = []
                    for x in idxs:
                        if isinstance(x, int) and 0 <= x < len(theme_names):
                            valid_idxs.append(x)
                        else:
                            valid_idxs.append(0)  # Default to first theme
                    assignments.extend(valid_idxs)
                    print(" ✅")
                    batch_success = True
                    break
                except Exception as e:
                    if batch_attempt == 0:
                        error_msg = str(e)
                        if len(error_msg) > 30:
                            error_msg = error_msg[:30] + "..."
                        print(f" ⚠️  Error ({error_msg})", end="", flush=True)
                        continue  # Try again
                    else:
                        error_msg = str(e)
                        if len(error_msg) > 50:
                            error_msg = error_msg[:50] + "..."
                        print(f" ⚠️  Failed ({error_msg}) - using theme 0")
                        batch_success = False
                        break

            if not batch_success:
                assignments.extend([0] * len(batch))

        print(
            f"      ✅ Theme assignment completed in {(time.time() - start_time) / 60:.1f} minutes"
        )
        return assignments

    def _normalize(self, s: str) -> str:
        return re.sub(r"\s+", " ", (s or "")).strip().lower()

    def _pick_representative_quote(self, texts: List[str]) -> str:
        if not texts:
            return ""
        lengths = [len(t) for t in texts]
        median_len = sorted(lengths)[len(lengths) // 2]
        best = min(texts, key=lambda t: abs(len(t) - median_len))
        return best

    def analyze_themes_for_question(
        self, question_column: str, satisfaction_segment: Optional[str] = None
    ):
        db_col = self._db_col_for_question(question_column)
        if not db_col:
            print(f"❌ Unknown question column: {question_column}")
            return

        # Check dataset size and use appropriate method
        cur = self.conn.cursor()
        count_query = f"SELECT COUNT(*) FROM survey_responses WHERE {db_col} IS NOT NULL AND TRIM({db_col}) != ''"
        params = []
        if satisfaction_segment:
            count_query += " AND satisfaction_rating = ?"
            params.append(satisfaction_segment)
        cur.execute(count_query, params)
        total_count = cur.fetchone()[0]

        large_dataset_threshold = self.config.get("large_dataset_threshold", 5000)

        if total_count > large_dataset_threshold:
            print(
                f"📊 Large dataset detected ({total_count:,} responses). Using memory-efficient processing..."
            )
            # For very large datasets, we'll need to modify the AI processing approach
            # For now, we'll still use the regular method but warn the user
            print(
                "⚠️  Note: AI processing with large datasets may take significant time and memory"
            )

        print(f"      📄 Fetching {total_count:,} responses from database...")
        rows = self._fetch_responses(db_col, satisfaction_segment)
        if not rows:
            print("⚠️  No responses to analyze")
            return

        ids, texts, names, companies, sats = zip(*rows)
        client = self._openai_client()

        print(f"      🤖 Generating themes using AI for {len(texts):,} responses...")
        if client:
            print(f"      🔗 Making API call to OpenAI...")
            proposed = self._ai_propose_themes(
                client, question_column, list(texts), satisfaction_segment
            )
            if not proposed:
                print(f"      ⚠️  AI returned no themes, using fallback")
                proposed = [
                    {
                        "name": "General Feedback",
                        "description": "Miscellaneous feedback grouped together.",
                        "representative_quote": texts[0],
                    }
                ]
        else:
            print(f"      ❌ No OpenAI client available, using fallback themes")
            proposed = [
                {
                    "name": "General Feedback",
                    "description": "Miscellaneous feedback grouped together.",
                    "representative_quote": texts[0],
                }
            ]

        print(
            f"      📝 AI generated {len(proposed)} themes: {[t['name'] for t in proposed]}"
        )

        theme_names = [t["name"] for t in proposed]
        theme_descs = {t["name"]: t["description"] for t in proposed}

        print(f"      🏷️  Assigning {len(texts):,} responses to themes...")
        if client and theme_names:
            assignments = self._ai_assign_themes(
                client, question_column, list(texts), theme_names
            )
        else:
            assignments = [0] * len(texts)
        print(f"      ✅ Theme assignments completed")

        cur = self.conn.cursor()
        if satisfaction_segment:
            cur.execute(
                "DELETE FROM themes WHERE question_column=? AND satisfaction_segment=?",
                (question_column, satisfaction_segment),
            )
            cur.execute(
                "DELETE FROM theme_assignments WHERE question_column=? AND satisfaction_segment=?",
                (question_column, satisfaction_segment),
            )
        else:
            cur.execute(
                "DELETE FROM themes WHERE question_column=? AND satisfaction_segment IS NULL",
                (question_column,),
            )
            cur.execute(
                "DELETE FROM theme_assignments WHERE question_column=? AND satisfaction_segment IS NULL",
                (question_column,),
            )

        for r_id, theme_idx in zip(ids, assignments):
            # Validate theme_idx is within range
            if theme_names and 0 <= theme_idx < len(theme_names):
                nm = theme_names[theme_idx]
            else:
                nm = "General Feedback"
                if theme_names and theme_idx >= len(theme_names):
                    print(
                        f"⚠️  Warning: Theme index {theme_idx} out of range (max: {len(theme_names) - 1}), using 'General Feedback'"
                    )
            cur.execute(
                "INSERT INTO theme_assignments (response_id, question_column, theme_name, satisfaction_segment, confidence) VALUES (?, ?, ?, ?, ?)",
                (int(r_id), question_column, nm, satisfaction_segment, 1.0),
            )
        self.conn.commit()

        theme_to_texts: Dict[str, List[str]] = {}
        for theme_idx, tx in zip(assignments, texts):
            # Validate theme_idx is within range
            if theme_names and 0 <= theme_idx < len(theme_names):
                nm = theme_names[theme_idx]
            else:
                nm = "General Feedback"
            theme_to_texts.setdefault(nm, []).append(tx)

        for nm in theme_names or ["General Feedback"]:
            t_texts = theme_to_texts.get(nm, [])
            rep_quote = self._pick_representative_quote(t_texts)
            respondent, company = self._find_quote_attribution(
                rep_quote, question_column, satisfaction_segment
            )
            cur.execute(
                "INSERT INTO themes (question_column, theme_name, theme_description, satisfaction_segment, representative_quote, representative_respondent, representative_company) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    question_column,
                    nm,
                    theme_descs.get(nm, ""),
                    satisfaction_segment,
                    rep_quote,
                    respondent,
                    company,
                ),
            )
        self.conn.commit()

        print(
            f"✅ Saved themes and assignments for: {question_column}{' [' + satisfaction_segment + ']' if satisfaction_segment else ''}"
        )

    def _find_quote_attribution(
        self, quote: str, question_column: str, satisfaction_segment: Optional[str]
    ) -> Tuple[str, str]:
        db_col = self._db_col_for_question(question_column)
        if not db_col or not quote:
            return ("Anonymous", "Company not provided")
        cur = self.conn.cursor()

        cur.execute(
            f"SELECT contact_name, contact_company, {db_col} FROM survey_responses WHERE {db_col} IS NOT NULL"
            + (" AND satisfaction_rating = ?" if satisfaction_segment else ""),
            ((satisfaction_segment,) if satisfaction_segment else ()),
        )
        for name, company, text in cur.fetchall():
            if self._normalize(text) == self._normalize(quote):
                return (name or "Anonymous", company or "Company not provided")

        like_pat = f"%{quote[:30]}%"
        base = f"SELECT contact_name, contact_company FROM survey_responses WHERE {db_col} LIKE ?"
        params = [like_pat]
        if satisfaction_segment:
            base += " AND satisfaction_rating = ?"
            params.append(satisfaction_segment)
        cur.execute(base, params)
        r = cur.fetchone()
        if r:
            return (r[0] or "Anonymous", r[1] or "Company not provided")
        return ("Anonymous", "Company not provided")

    def _group_satisfaction_level(self, satisfaction_rating: str) -> str:
        """Group satisfaction ratings into 3 main categories"""
        if not satisfaction_rating:
            return "Unknown"

        rating = satisfaction_rating.lower().strip()

        if "very satisfied" in rating or rating == "satisfied":
            return "Satisfied"
        elif "dissatisfied" in rating or "very dissatisfied" in rating:
            return "Dissatisfied"
        elif "neither" in rating or "neutral" in rating:
            return "Neutral"
        else:
            return "Unknown"

    def run_theme_analysis(self):
        start_time = time.time()
        print(f"\n🧠 Starting AI theme analysis...")

        # First, update satisfaction ratings to grouped categories
        print("📊 Grouping satisfaction levels into 3 categories...")
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT satisfaction_rating FROM survey_responses WHERE satisfaction_rating IS NOT NULL"
        )
        original_levels = [r[0] for r in cur.fetchall()]

        # Update all satisfaction ratings to grouped versions
        for original_level in original_levels:
            grouped_level = self._group_satisfaction_level(original_level)
            cur.execute(
                "UPDATE survey_responses SET satisfaction_rating = ? WHERE satisfaction_rating = ?",
                (grouped_level, original_level),
            )
        self.conn.commit()

        # Get the grouped satisfaction levels
        cur.execute(
            "SELECT DISTINCT satisfaction_rating FROM survey_responses WHERE satisfaction_rating IS NOT NULL"
        )
        levels = [r[0] for r in cur.fetchall()]
        print(
            f"✅ Grouped into {len(levels)} satisfaction categories: {', '.join(levels)}"
        )

        # Calculate total analysis tasks
        total_tasks = len(levels) + len(self.OPEN_TEXT_COLUMNS[1:])
        completed_tasks = 0

        main_reason = self.OPEN_TEXT_COLUMNS[0]
        print(
            f"\n📊 Analyzing '{main_reason}' by satisfaction segment ({len(levels)} segments)..."
        )

        for i, level in enumerate(levels, 1):
            print(f"   🔄 Processing satisfaction level '{level}' ({i}/{len(levels)})")
            self.analyze_themes_for_question(main_reason, level)
            completed_tasks += 1
            overall_progress = completed_tasks / total_tasks * 100
            print(
                f"   ✅ Completed '{level}' | Overall Progress: {overall_progress:.1f}%"
            )

        print(
            f"\n🔍 Analyzing remaining {len(self.OPEN_TEXT_COLUMNS[1:])} questions (all satisfaction levels combined)..."
        )
        for i, q in enumerate(self.OPEN_TEXT_COLUMNS[1:], 1):
            print(
                f"   🔄 Processing question {i}/{len(self.OPEN_TEXT_COLUMNS[1:])}: '{q[:50]}{'...' if len(q) > 50 else ''}'"
            )
            self.analyze_themes_for_question(q, satisfaction_segment=None)
            completed_tasks += 1
            overall_progress = completed_tasks / total_tasks * 100
            print(
                f"   ✅ Completed question {i} | Overall Progress: {overall_progress:.1f}%"
            )

        total_time = time.time() - start_time
        print(
            f"\n🎉 Theme analysis complete! Processed {total_tasks} analyses in {total_time / 60:.1f} minutes"
        )

    def _counts_df(
        self, question: str, satisfaction_segment: Optional[str] = None
    ) -> pd.DataFrame:
        cur = self.conn.cursor()
        if satisfaction_segment:
            cur.execute(
                "SELECT theme_name, COUNT(*) as cnt FROM theme_assignments WHERE question_column=? AND satisfaction_segment=? GROUP BY theme_name ORDER BY cnt DESC",
                (question, satisfaction_segment),
            )
        else:
            cur.execute(
                "SELECT theme_name, COUNT(*) as cnt FROM theme_assignments WHERE question_column=? AND (satisfaction_segment IS NULL OR satisfaction_segment='') GROUP BY theme_name ORDER BY cnt DESC",
                (question,),
            )
        rows = cur.fetchall()
        return pd.DataFrame(rows, columns=["theme_name", "count"])

    def _ensure_dir(self, path: str):
        os.makedirs(path, exist_ok=True)

    def generate_charts(self, out_dir: str = "reports"):
        print(f"\n📊 Generating charts and visualizations...")
        start_time = time.time()

        self._ensure_dir(out_dir)
        print(f"   📁 Created output directory: {out_dir}")

        q = self.OPEN_TEXT_COLUMNS[0]
        cur = self.conn.cursor()
        cur.execute(
            "SELECT DISTINCT satisfaction_segment FROM theme_assignments WHERE question_column=?",
            (q,),
        )
        segs = [r[0] for r in cur.fetchall() if r[0]]

        # Generate stacked bar chart for main question
        print(
            f"   📈 Creating stacked satisfaction chart for '{q[:50]}{'...' if len(q) > 50 else ''}'"
        )
        frames = []
        for seg in segs:
            df = self._counts_df(q, seg)
            if df.empty:
                continue
            df = df.set_index("theme_name").rename(columns={"count": seg})
            frames.append(df)
        if frames:
            merged = pd.concat(frames, axis=1).fillna(0).astype(int)
            merged = merged.sort_values(by=merged.columns.tolist(), ascending=False)
            plt.figure()
            merged.plot(kind="bar", stacked=True)
            plt.title("Main Reason for Satisfaction — Themes (stacked by satisfaction)")
            plt.tight_layout()
            path = os.path.join(out_dir, f"{slugify(q)}_stacked.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"   ✅ Saved: {path}")

        # Generate individual charts for other questions
        total_questions = len(self.OPEN_TEXT_COLUMNS[1:])
        for i, q in enumerate(self.OPEN_TEXT_COLUMNS[1:], 1):
            print(
                f"   📊 Creating chart {i}/{total_questions}: '{q[:50]}{'...' if len(q) > 50 else ''}'"
            )
            df = self._counts_df(q, satisfaction_segment=None)
            if df.empty:
                print(f"   ⚠️  No data for '{q[:30]}...', skipping chart")
                continue
            df = df.sort_values("count", ascending=False).head(12)
            plt.figure()
            df.plot(kind="bar", x="theme_name", y="count", legend=False)
            plt.title(f"Top Themes — {q}")
            plt.tight_layout()
            path = os.path.join(out_dir, f"{slugify(q)}_top.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"   ✅ Saved: {path}")

        total_time = time.time() - start_time
        charts_created = 1 + total_questions  # stacked chart + individual charts
        print(
            f"\n🎨 Chart generation complete! Created {charts_created} charts in {total_time:.1f} seconds"
        )

    def generate_reports(self, output_file: str = "cpm_survey_analysis.md"):
        print(f"\n📄 Generating comprehensive markdown report...")
        start_time = time.time()

        cur = self.conn.cursor()

        print("   📊 Gathering survey statistics...")
        cur.execute("SELECT COUNT(*) FROM survey_responses")
        total = cur.fetchone()[0]

        cur.execute(
            "SELECT satisfaction_rating, COUNT(*) FROM survey_responses WHERE satisfaction_rating IS NOT NULL GROUP BY satisfaction_rating"
        )
        sat_rows = cur.fetchall()

        print("   📝 Writing report structure...")
        md = []
        md.append("# CPM Customer Survey Analysis Report\n")
        md.append(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        md.append("## Executive Summary\n")
        md.append(f"- **Total Responses Analyzed**: {total:,}\n")
        md.append(f"- **Analysis Date**: {datetime.now().strftime('%B %d, %Y')}\n")

        md.append("\n## Satisfaction Distribution\n")
        md.append("| Satisfaction Level | Count |\n|---|---|\n")
        for s, c in sat_rows:
            md.append(f"| {s or 'Unknown'} | {c} |\n")

        print("   🔍 Processing theme analysis results...")
        md.append("\n## Theme Analysis by Question\n")
        for i, q in enumerate(self.OPEN_TEXT_COLUMNS, 1):
            print(
                f"   📋 Writing analysis for question {i}/{len(self.OPEN_TEXT_COLUMNS)}: '{q[:40]}{'...' if len(q) > 40 else ''}'",
                end="\r",
            )
            md.append(f"\n### {q}\n")
            if q == self.OPEN_TEXT_COLUMNS[0]:
                cur.execute(
                    "SELECT DISTINCT satisfaction_segment FROM theme_assignments WHERE question_column=?",
                    (q,),
                )
                segs = [r[0] for r in cur.fetchall() if r[0]]
                for seg in segs:
                    md.append(f"\n#### Satisfaction Level: {seg}\n")
                    cur.execute(
                        "SELECT theme_name, COUNT(*) FROM theme_assignments WHERE question_column=? AND satisfaction_segment=? GROUP BY theme_name ORDER BY COUNT(*) DESC",
                        (q, seg),
                    )
                    themes_data = cur.fetchall()

                    # Get quotes for all themes at once
                    cur.execute(
                        "SELECT theme_name, representative_quote, representative_respondent, representative_company FROM themes WHERE question_column=? AND satisfaction_segment=?",
                        (q, seg),
                    )
                    quotes_data = {
                        name: (quote, person, comp)
                        for name, quote, person, comp in cur.fetchall()
                    }

                    # Write each theme with its quote immediately after
                    for name, cnt in themes_data:
                        md.append(f"- **{name}** — {cnt}\n")
                        if name in quotes_data and quotes_data[name][0]:
                            quote, person, comp = quotes_data[name]
                            md.append(f'  > "{quote}"\n')
                            who = person or "Anonymous"
                            comp2 = f", {comp}" if comp else ""
                            md.append(f"  > *— {who}{comp2}*\n")
            else:
                md.append("\n")
                cur.execute(
                    "SELECT theme_name, COUNT(*) FROM theme_assignments WHERE question_column=? AND (satisfaction_segment IS NULL OR satisfaction_segment='') GROUP BY theme_name ORDER BY COUNT(*) DESC",
                    (q,),
                )
                themes_data = cur.fetchall()

                # Get quotes for all themes at once
                cur.execute(
                    "SELECT theme_name, representative_quote, representative_respondent, representative_company FROM themes WHERE question_column=? AND (satisfaction_segment IS NULL OR satisfaction_segment='')",
                    (q,),
                )
                quotes_data = {
                    name: (quote, person, comp)
                    for name, quote, person, comp in cur.fetchall()
                }

                # Write each theme with its quote immediately after
                for name, cnt in themes_data:
                    md.append(f"- **{name}** — {cnt}\n")
                    if name in quotes_data and quotes_data[name][0]:
                        quote, person, comp = quotes_data[name]
                        md.append(f'  > "{quote}"\n')
                        who = person or "Anonymous"
                        comp2 = f", {comp}" if comp else ""
                        md.append(f"  > *— {who}{comp2}*\n")

        print(f"\n   💾 Writing report to file: {output_file}")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("".join(md))

        total_time = time.time() - start_time
        file_size = os.path.getsize(output_file) / 1024  # KB
        print(
            f"✅ Report complete! Generated {output_file} ({file_size:.1f} KB) in {total_time:.1f} seconds"
        )

    def run_full_analysis(self):
        print("🎯 Starting Full CPM Survey Analysis")
        print("=" * 50)
        self.setup_database()
        if not self.import_csv_data():
            print("❌ Import failed")
            return
        self.detect_and_translate_responses()
        self.run_theme_analysis()
        self.generate_charts("reports")
        self.generate_reports("cpm_survey_analysis.md")
        print("\n🎉 Full Analysis Complete!")

    def show_database_summary(self):
        cur = self.conn.cursor()
        print("\n🗄️  DATABASE SUMMARY")
        cur.execute("SELECT COUNT(*) FROM survey_responses")
        total = cur.fetchone()[0]
        print(f"📊 Total Responses: {total}")
        cur.execute(
            "SELECT satisfaction_rating, COUNT(*) FROM survey_responses WHERE satisfaction_rating IS NOT NULL GROUP BY satisfaction_rating ORDER BY COUNT(*) DESC"
        )
        print("\n📈 Satisfaction Distribution:")
        for s, c in cur.fetchall():
            print(f"   • {s}: {c}")
        cur.execute("SELECT COUNT(*) FROM themes")
        print(f"\n🎨 Themes Identified: {cur.fetchone()[0]}")
        cur.execute("SELECT COUNT(*) FROM theme_assignments")
        print(f"🧩 Theme Assignments: {cur.fetchone()[0]}")

    def clear_database(self):
        cur = self.conn.cursor()
        cur.execute("DELETE FROM survey_responses")
        cur.execute("DELETE FROM themes")
        cur.execute("DELETE FROM theme_assignments")
        cur.execute("DELETE FROM analysis_runs")
        self.conn.commit()


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="CPM Survey Text Analysis Tool (Refactored)"
    )
    parser.add_argument("--csv", help="Path to survey CSV")
    parser.add_argument("--db", default="cpm_survey.db", help="Path to SQLite DB file")
    parser.add_argument("--full", action="store_true", help="Run full pipeline")
    parser.add_argument(
        "--translate-only", action="store_true", help="Translate step only"
    )
    parser.add_argument(
        "--analyze-only", action="store_true", help="Theme analysis only"
    )
    parser.add_argument(
        "--reports-only", action="store_true", help="Generate charts/reports only"
    )
    parser.add_argument(
        "--assume-yes", action="store_true", help="Non-interactive mode (no prompts)"
    )
    parser.add_argument(
        "--db-action",
        choices=["replace", "append"],
        help="Database action: 'replace' clears existing data, 'append' adds new data only",
    )
    args = parser.parse_args()

    def find_csv_files():
        """Find all CSV files in current directory"""
        import glob

        return glob.glob("*.csv")

    def prompt_csv_selection(csv_files):
        """Interactive CSV file selection menu"""
        print(f"\n📁 Found {len(csv_files)} CSV file(s) in current directory:")
        for i, file in enumerate(csv_files, 1):
            size = os.path.getsize(file) / 1024  # KB
            print(f"  {i}. {file} ({size:.1f} KB)")

        print("  0. Enter custom path")

        while True:
            try:
                choice = input("\nSelect CSV file (number): ").strip()
                if choice == "0":
                    custom_path = input("Enter CSV file path: ").strip()
                    if custom_path and os.path.isfile(custom_path):
                        return custom_path
                    print("❌ File not found. Please try again.")
                    continue

                choice_num = int(choice)
                if 1 <= choice_num <= len(csv_files):
                    return csv_files[choice_num - 1]
                else:
                    print(f"❌ Please enter a number between 0 and {len(csv_files)}")
            except (ValueError, EOFError, KeyboardInterrupt):
                print("\n❌ Operation cancelled or invalid input.")
                return None

    # Get CSV file path - discover and prompt if not provided
    csv_file = args.csv
    if not csv_file:
        if args.assume_yes or os.getenv("ASSUME_YES") == "1":
            print("❌ CSV file required in non-interactive mode. Use --csv argument.")
            return

        # Look for CSV files in current directory
        csv_files = find_csv_files()

        if csv_files:
            csv_file = prompt_csv_selection(csv_files)
            if not csv_file:
                print("\n❌ Operation cancelled.")
                return
        else:
            print("📁 No CSV files found in current directory.")
            print("Please provide the path to your survey CSV file.")
            while True:
                try:
                    csv_file = input("Enter CSV file path: ").strip()
                    if csv_file:
                        break
                    print("❌ Please enter a valid file path.")
                except (EOFError, KeyboardInterrupt):
                    print("\n❌ Operation cancelled.")
                    return

    # Validate CSV file exists
    if not os.path.isfile(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    # OpenAI model selection
    ai_model = "gpt-5-mini"  # Default
    if not (args.assume_yes or os.getenv("ASSUME_YES") == "1"):
        print("\n🤖 Select OpenAI model for analysis:")
        model_options = [
            (
                "gpt-5-mini",
                "GPT-5 Mini (Default) - Cost-optimized reasoning, balances speed & capability",
            ),
            (
                "gpt-5",
                "GPT-5 - Most intelligent, complex reasoning & broad world knowledge",
            ),
            (
                "gpt-5-nano",
                "GPT-5 Nano - High-throughput, simple instruction-following",
            ),
            ("gpt-4o", "GPT-4o - Previous generation, very capable"),
            (
                "gpt-4o-mini",
                "GPT-4o Mini - Previous generation, fast and cost-effective",
            ),
            ("gpt-4-turbo", "GPT-4 Turbo - Reliable previous generation"),
        ]

        for i, (model, description) in enumerate(model_options, 1):
            default_marker = " (Default)" if model == "gpt-5-mini" else ""
            print(f"   {i}. {description}{default_marker}")

        try:
            choice = input(
                f"Enter choice (1-{len(model_options)}) or press Enter for default: "
            ).strip()
            if choice and choice.isdigit():
                choice_idx = int(choice) - 1
                if 0 <= choice_idx < len(model_options):
                    ai_model = model_options[choice_idx][0]
                    print(f"✅ Selected: {model_options[choice_idx][1]}")
                else:
                    print(f"✅ Using default: GPT-5 Mini")
            else:
                print(f"✅ Using default: GPT-5 Mini")
        except (EOFError, KeyboardInterrupt):
            print(f"\n✅ Using default: GPT-5 Mini")

    # Note about GPT-5 models
    if ai_model.startswith("gpt-5"):
        print(
            f"ℹ️  Note: {ai_model} uses Chat Completions API. For optimal performance, GPT-5 models work best with the new Responses API."
        )

    # Example configuration - users can customize these settings
    config = {
        "ai_model": ai_model,
        "ai_max_retries": 3,
        "ai_base_delay": 0.1,
        "min_responses": 10,
        "large_dataset_threshold": 5000,
        # Uncomment and modify these to match your CSV column names:
        # 'satisfaction_column': 'Your satisfaction column name',
        # 'name_column': 'Your name column',
        # 'email_column': 'Your email column',
        # 'company_column': 'Your company column',
    }

    analyzer = CPMSurveyAnalyzer(
        csv_file,
        db_file=args.db,
        assume_yes=args.assume_yes or os.getenv("ASSUME_YES") == "1",
        config=config,
    )

    # Pass db_action to analyzer
    if args.db_action:
        analyzer._db_action = args.db_action
    analyzer.setup_database()

    if args.full:
        analyzer.run_full_analysis()
        return

    if args.translate_only:
        if analyzer.import_csv_data():
            analyzer.detect_and_translate_responses()
        return

    if args.analyze_only:
        if analyzer.import_csv_data():
            analyzer.detect_and_translate_responses()
            analyzer.run_theme_analysis()
        return

    if args.reports_only:
        analyzer.generate_charts("reports")
        analyzer.generate_reports("cpm_survey_analysis.md")
        return

    analyzer.show_database_summary()


if __name__ == "__main__":
    main()
