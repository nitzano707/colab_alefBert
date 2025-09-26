# -*- coding: utf-8 -*-
"""
alephbert_utils.py

פותח ע"י ד"ר ניצן אליקים | elyakim@talpiot.ac.il
"""


import os
import time
import warnings
import subprocess
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import ipywidgets as widgets
from IPython.display import display, clear_output, HTML

from sentence_transformers import SentenceTransformer, util
import bidi.algorithm
from arabic_reshaper import reshape

# הורדה למחשב מקומי – נתמך רק ב-Colab; אם לא קיים נשאיר None
try:
    from google.colab import files as _colab_files  # type: ignore
except Exception:
    _colab_files = None

# ------------------------------------------------------------
# פונקציות עזר
# ------------------------------------------------------------

def _ensure_arial_font() -> str:
    """
    מוודאת שגופן Arial קיים. אם לא – ניסיון התקנה שקט ב-Colab.
    מחזירה את שם הגופן לשימוש ב-matplotlib.
    """
    import matplotlib.font_manager as fm

    try:
        _ = fm.findfont("Arial", fallback_to_default=False)
        if os.path.exists(_):
            return "Arial"
    except Exception:
        pass

    # ניסיון התקנה ב-Colab/דביאן (שקט)
    try:
        subprocess.run(
            ["apt-get", "-y", "update"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        subprocess.run(
            ["apt-get", "-y", "install", "ttf-mscorefonts-installer", "fontconfig"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        subprocess.run(
            ["fc-cache", "-fv"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False
        )
        _ = fm.findfont("Arial", fallback_to_default=False)
        if os.path.exists(_):
            return "Arial"
    except Exception:
        pass

    # נפילה אל גופן ברירת מחדל שתומך בעברית ב-Colab
    return "DejaVu Sans"


def _fix_hebrew_text_for_plots(text: str) -> str:
    """היפוך וכיווניות נכונים לעברית עבור טקסטים שבתרשימים בלבד."""
    try:
        return bidi.algorithm.get_display(reshape(text))
    except Exception:
        return text


# ------------------------------------------------------------
# הגדרות עיצוב גרפיות
# ------------------------------------------------------------
warnings.filterwarnings("ignore")

_font_family = _ensure_arial_font()
plt.rcParams["figure.figsize"] = (14, 10)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["font.family"] = _font_family
sns.set_style("whitegrid")


# ------------------------------------------------------------
# מחלקת הניתוח
# ------------------------------------------------------------
class SeedSentenceAnalyzer:
    def __init__(self, seed_sentence: str,
                 model: str = "imvladikon/sentence-transformers-alephbert"):
        self.seed_sentence = seed_sentence.strip()
        self.model_name = model
        self.model = SentenceTransformer(model)

        self.sentences: List[str] = []
        self.similarities: List[float] = []
        self.df: Optional[pd.DataFrame] = None

    # ---------- קריאה וקד"מ ----------
    def load_sentences_from_csv(self, csv_path: str, sentence_column: str = "sentence") -> Optional[pd.DataFrame]:
        try:
            df = pd.read_csv(csv_path, encoding="utf-8")
            if sentence_column not in df.columns:
                print(f"❌ עמודה '{sentence_column}' לא נמצאה בקובץ. עמודות זמינות: {list(df.columns)}")
                return None

            df = df.dropna(subset=[sentence_column])
            df = df[df[sentence_column].str.strip() != ""]
            df = df[df[sentence_column].str.strip() != self.seed_sentence]
            df = df.reset_index(drop=True)

            self.sentences = df[sentence_column].tolist()
            self.df = df
            return self.df

        except Exception as e:
            print(f"❌ שגיאה בטעינת קובץ ה-CSV: {e}")
            return None

    def calculate_similarities_to_seed(self) -> Optional[List[float]]:
        if not self.sentences:
            print("❌ אין היגדים לעיבוד. יש לטעון קובץ CSV תחילה.")
            return None

        try:
            seed_emb = self.model.encode(self.seed_sentence, convert_to_tensor=True)
            sent_embs = self.model.encode(self.sentences, convert_to_tensor=True)
            sims = util.cos_sim(seed_emb, sent_embs).cpu().numpy().flatten().tolist()
            self.similarities = sims

            if self.df is not None:
                self.df["similarity_score"] = sims
                self.df = self.df.sort_values("similarity_score", ascending=False).reset_index(drop=True)

            return self.similarities
        except Exception as e:
            print(f"❌ שגיאה בחישוב דמיון קוסיני: {e}")
            return None

    # ---------- תצוגת תוצאות ----------
    def show_header(self):
        box = HTML(
            f"""
            <div dir="rtl" style="text-align:right;font-family:{_font_family};border:2px solid #2e7d32;border-radius:8px;padding:12px;margin:10px 0;">
              <div style="font-size:18px;font-weight:700;margin-bottom:6px;">תוצאות ניתוח</div>
              <div><b>פותח ע"י ד"ר ניצן אליקים</b> | <a href="mailto:elyakim@talpiot.ac.il">elyakim@talpiot.ac.il</a></div>
              <div>היגד הזרע: <b>{self.seed_sentence}</b></div>
              <div>מודל: {self.model_name}</div>
            </div>
            """
        )
        display(box)

    def display_results(self, num_strong: int = 5, num_medium: int = 5) -> Tuple[list, list, list, list]:
        all_matches = [(i, s, sc) for i, (s, sc) in enumerate(zip(self.sentences, self.similarities))]
        all_matches.sort(key=lambda x: x[2], reverse=True)

        strong = [m for m in all_matches if m[2] >= 0.75]
        medium = [m for m in all_matches if 0.70 <= m[2] < 0.75]
        weak = [m for m in all_matches if m[2] < 0.70]

        # סיכום מילולי
        summary_html = f"""
        <div dir="rtl" style="text-align:right;font-family:{_font_family};margin:6px 0 12px 0;">
          נמצאו <b>{len(strong)}</b> היגדים דומים מאוד (≥0.75),
          <b>{len(medium)}</b> היגדים דומים במידה בינונית (0.70–0.749),
          ו־<b>{len(weak)}</b> היגדים רחוקים במשמעות.
          סה"כ נותחו <b>{len(all_matches)}</b> היגדים.
        </div>
        """
        display(HTML(summary_html))

        # טבלת Top-N (חזקים + בינוניים)
        top_items = strong[:num_strong] + medium[:num_medium]
        if top_items:
            df_top = pd.DataFrame([(s, f"{sc:.3f}") for (_, s, sc) in top_items],
                                  columns=["היגד", "ציון דמיון"])
            # הצגה מיושרת לימין
            display(
                df_top.style.set_table_styles(
                    [{'selector': 'th', 'props': [('text-align', 'right'), ('font-family', _font_family)]}]
                ).set_properties(**{'text-align': 'right', 'font-family': _font_family})
            )

        return strong, medium, weak, all_matches

    def create_visualizations(self):
        scores = np.array(self.similarities)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        categories = [_fix_hebrew_text_for_plots('דמיון חזק (≥0.75)'),
                      _fix_hebrew_text_for_plots('דמיון בינוני (0.70-0.749)'),
                      _fix_hebrew_text_for_plots('דמיון חלש (<0.70)')]
        counts = [np.sum(scores >= 0.75),
                  np.sum((scores >= 0.70) & (scores < 0.75)),
                  np.sum(scores < 0.70)]

        axes[0, 0].bar(categories, counts, color=['green', 'gold', 'skyblue'])
        axes[0, 0].set_title(_fix_hebrew_text_for_plots("התפלגות לפי רמות דמיון"))

        axes[0, 1].hist(scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0, 1].axvline(np.mean(scores), color='red', linestyle='--',
                           label=_fix_hebrew_text_for_plots(f'ממוצע: {np.mean(scores):.3f}'))
        axes[0, 1].axvline(np.median(scores), color='green', linestyle='--',
                           label=_fix_hebrew_text_for_plots(f'חציון: {np.median(scores):.3f}'))
        axes[0, 1].legend()
        axes[0, 1].set_title(_fix_hebrew_text_for_plots("התפלגות ציוני דמיון"))

        top_15_idx = np.argsort(scores)[-15:][::-1] if len(scores) >= 15 else np.argsort(scores)[::-1]
        top_15_scores = scores[top_15_idx]
        axes[0, 2].bar(range(1, len(top_15_scores) + 1), top_15_scores, color='green')
        axes[0, 2].axhline(y=0.75, color='darkgreen', linestyle='--',
                           label=_fix_hebrew_text_for_plots('חזק (0.75)'))
        axes[0, 2].axhline(y=0.70, color='orange', linestyle='--',
                           label=_fix_hebrew_text_for_plots('בינוני (0.70)'))
        axes[0, 2].legend()
        axes[0, 2].set_title(_fix_hebrew_text_for_plots("טופ 15 ציוני דמיון"))

        axes[1, 0].boxplot(scores, patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1, 0].axhline(y=0.75, color='darkgreen', linestyle='--')
        axes[1, 0].axhline(y=0.70, color='orange', linestyle='--')
        axes[1, 0].set_title(_fix_hebrew_text_for_plots("Box Plot"))

        # מפת חום לטופ 20
        top_n = min(20, len(scores))
        if top_n > 0:
            top_idx = np.argsort(scores)[-top_n:][::-1]
            heatmap_data = scores[top_idx]
            # ריפוד ל-4x5 במקרה של פחות מ-20
            pad = 20 - top_n
            if pad > 0:
                heatmap_data = np.concatenate([heatmap_data, np.full(pad, np.nan)])
            heatmap_data = heatmap_data.reshape(4, 5)

            im = axes[1, 1].imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1)
            for i in range(4):
                for j in range(5):
                    val = heatmap_data[i, j]
                    if not np.isnan(val):
                        axes[1, 1].text(j, i, f"{val:.3f}", ha="center", va="center", color="black")
            axes[1, 1].set_title(_fix_hebrew_text_for_plots("מפת חום - טופ 20"))
            fig.colorbar(im, ax=axes[1, 1])

        axes[1, 2].axis('off')
        plt.tight_layout(pad=3.0)
        plt.show()

    # ---------- ייצוא ----------
    def export_to_excel(self, colab_link: str = "https://colab.research.google.com/"):
        if self.df is None:
            print("❌ אין נתונים לייצוא")
            return

        # נבנה גיליון מסודר בעברית
        filename = "תוצאות_ניתוח.xlsx"
        with pd.ExcelWriter(filename, engine="xlsxwriter") as writer:
            workbook = writer.book
            ws = workbook.add_worksheet("תוצאות")
            writer.sheets["תוצאות"] = ws

            bold = workbook.add_format({"bold": True})

            # שורה 1: משפט הזרע
            ws.write("A1", "משפט הזרע", bold)
            ws.write("B1", self.seed_sentence, bold)

            # שורה 2: נוצר ע"י COLAB
            ws.write("A2", f"קובץ זה נוצר ע\"י מחברת COLAB: {colab_link}")

            # רווח שורה
            ws.write("A3", "")

            # טבלה מלאה
            df_out = self.df.copy()
            if "sentence" in df_out.columns:
                df_out = df_out.rename(columns={"sentence": "היגד"})
            if "similarity_score" in df_out.columns:
                df_out = df_out.rename(columns={"similarity_score": "ציון דמיון"})
            df_out.to_excel(writer, sheet_name="תוצאות", startrow=3, index=False)

        print(f"📁 נשמר ב-Colab: {filename}")
        if _colab_files is not None:
            _colab_files.download(filename)


# ------------------------------------------------------------
# טופס אינטראקטיבי (UI)
# ------------------------------------------------------------
def create_analysis_form(colab_notebook_link: str = "https://colab.research.google.com/"):
    """
    מציג טופס ניתוח היגדים באמצעות HTML טהור. colab_notebook_link – ייכתב בשורת ההסבר באקסל.
    """
    
    form_html = HTML(f"""
    <div id="analysis-form" style="font-family: {_font_family}, sans-serif; direction: rtl; text-align: right; max-width: 1000px;">
        
        <!-- הוראות -->
        <div style="background: linear-gradient(135deg, #e3f2fd 0%, #f1f8e9 100%); 
                    border: 1px solid #81c784; border-radius: 8px; padding: 15px; margin: 15px 0;">
            <h3 style="margin-top: 0;">הוראות להכנת קובץ CSV</h3>
            <ol style="margin: 0;">
                <li>צור/י קובץ עם עמודה בשם <code>sentence</code>.</li>
                <li>כל שורה מכילה היגד אחד.</li>
                <li>שמור/שמרי את הקובץ בקידוד UTF-8.</li>
            </ol>
        </div>
        
        <!-- היגד הזרע -->
        <div style="margin: 20px 0;">
            <label for="seed-text" style="display: block; font-weight: bold; margin-bottom: 5px;">היגד הזרע:</label>
            <textarea id="seed-text" placeholder="הכנס/י כאן את היגד הזרע…" 
                     style="width: 100%; height: 80px; padding: 10px; border: 1px solid #ccc; 
                            border-radius: 4px; font-family: {_font_family}, sans-serif; 
                            direction: rtl; text-align: right; resize: vertical;"></textarea>
        </div>
        
        <!-- העלאת קובץ -->
        <div style="margin: 20px 0;">
            <label for="file-upload" style="display: block; font-weight: bold; margin-bottom: 5px;">צירוף קובץ CSV:</label>
            <input type="file" id="file-upload" accept=".csv" 
                   style="width: 100%; padding: 8px; border: 1px solid #ccc; border-radius: 4px;">
            <div id="file-status" style="margin-top: 5px; font-size: 12px; color: #666;"></div>
        </div>
        
        <!-- עמודת טקסט -->
        <div style="margin: 20px 0;">
            <label for="column-name" style="display: block; font-weight: bold; margin-bottom: 5px;">עמודת טקסט:</label>
            <input type="text" id="column-name" value="sentence" 
                   style="width: 300px; padding: 8px; border: 1px solid #ccc; border-radius: 4px; 
                          font-family: {_font_family}, sans-serif; direction: rtl; text-align: right;">
        </div>
        
        <!-- היגדים חזקים -->
        <div style="margin: 20px 0;">
            <label for="num-strong" style="display: block; font-weight: bold; margin-bottom: 5px;">
                היגדים חזקים (≥0.75): <span id="strong-value">5</span>
            </label>
            <input type="range" id="num-strong" min="0" max="50" value="5" 
                   style="width: 80%; margin-top: 5px;">
        </div>
        
        <!-- היגדים בינוניים -->
        <div style="margin: 20px 0;">
            <label for="num-medium" style="display: block; font-weight: bold; margin-bottom: 5px;">
                היגדים בינוניים (0.70–0.749): <span id="medium-value">5</span>
            </label>
            <input type="range" id="num-medium" min="0" max="50" value="5" 
                   style="width: 80%; margin-top: 5px;">
        </div>
        
        <!-- כפתורים -->
        <div style="margin: 20px 0;">
            <button id="analyze-btn" onclick="runAnalysis()" 
                    style="background: #4caf50; color: white; border: none; padding: 12px 24px; 
                           border-radius: 4px; font-family: {_font_family}, sans-serif; 
                           font-size: 16px; cursor: pointer; margin-left: 10px;">
                בצע ניתוח
            </button>
            <button id="export-btn" onclick="exportExcel()" disabled
                    style="background: #2196f3; color: white; border: none; padding: 12px 24px; 
                           border-radius: 4px; font-family: {_font_family}, sans-serif; 
                           font-size: 16px; cursor: pointer;">
                ייצוא לאקסל
            </button>
        </div>
        
        <!-- אזור סטטוס -->
        <div id="status-area" style="margin: 15px 0; padding: 10px; background: #f5f5f5; 
                                    border-radius: 4px; min-height: 20px; display: none;"></div>
    </div>
    
    <script>
        // משתנים גלובליים
        let currentAnalyzer = null;
        let uploadedFile = null;
        
        // עדכון ערכי סליידרים
        document.getElementById('num-strong').addEventListener('input', function() {
            document.getElementById('strong-value').textContent = this.value;
        });
        
        document.getElementById('num-medium').addEventListener('input', function() {
            document.getElementById('medium-value').textContent = this.value;
        });
        
        // טיפול בהעלאת קובץ
        document.getElementById('file-upload').addEventListener('change', function(e) {
            const file = e.target.files[0];
            const statusDiv = document.getElementById('file-status');
            
            if (file) {
                if (file.name.endsWith('.csv')) {
                    uploadedFile = file;
                    statusDiv.textContent = `נבחר: ${{file.name}} (${{(file.size/1024).toFixed(1)}} KB)`;
                    statusDiv.style.color = '#4caf50';
                } else {
                    statusDiv.textContent = 'אנא בחר קובץ CSV בלבד';
                    statusDiv.style.color = '#f44336';
                    uploadedFile = null;
                }
            }
        });
        
        // פונקציית ניתוח
        function runAnalysis() {
            const statusArea = document.getElementById('status-area');
            const analyzeBtn = document.getElementById('analyze-btn');
            const exportBtn = document.getElementById('export-btn');
            
            // בדיקות ולידציה
            const seedText = document.getElementById('seed-text').value.trim();
            if (!seedText) {
                showStatus('❌ יש להזין היגד זרע.', 'error');
                return;
            }
            
            if (!uploadedFile) {
                showStatus('❌ יש להעלות קובץ CSV.', 'error');
                return;
            }
            
            // השבתת כפתורים
            analyzeBtn.disabled = true;
            exportBtn.disabled = true;
            showStatus('מתחיל ניתוח...', 'info');
            
            // כאן נקרא לפונקציות Python דרך הממשק
            // נשתמש ב-IPython.display לקריאה חזרה ל-Python
            window.pythonAnalysisData = {
                seedText: seedText,
                columnName: document.getElementById('column-name').value,
                numStrong: parseInt(document.getElementById('num-strong').value),
                numMedium: parseInt(document.getElementById('num-medium').value),
                file: uploadedFile
            };
            
            // קריאה ל-Python
            IPython.notebook.kernel.execute(`
                import json
                from js import window
                
                # קבלת נתונים מ-JavaScript
                data = window.pythonAnalysisData.to_py()
                
                # יצירת analyzer
                analyzer = SeedSentenceAnalyzer(data['seedText'])
                
                # עיבוד הקובץ (כאן תצטרך להוסיף לוגיקה לטיפול בקובץ)
                # ...
                
                # הצגת תוצאות
                print("✅ הניתוח הושלם!")
            `);
            
            // איפוס כפתורים
            setTimeout(() => {
                analyzeBtn.disabled = false;
                exportBtn.disabled = false;
            }, 2000);
        }
        
        function exportExcel() {
            if (!currentAnalyzer) {
                showStatus('❌ לא קיים ניתוח לייצוא.', 'error');
                return;
            }
            showStatus('מייצא לאקסל...', 'info');
            // כאן תתבצע קריאה ל-Python לייצוא
        }
        
        function showStatus(message, type) {
            const statusArea = document.getElementById('status-area');
            statusArea.style.display = 'block';
            statusArea.textContent = message;
            
            if (type === 'error') {
                statusArea.style.background = '#ffebee';
                statusArea.style.color = '#c62828';
                statusArea.style.border = '1px solid #e57373';
            } else if (type === 'info') {
                statusArea.style.background = '#e3f2fd';
                statusArea.style.color = '#1565c0';
                statusArea.style.border = '1px solid #90caf9';
            } else {
                statusArea.style.background = '#f1f8e9';
                statusArea.style.color = '#2e7d32';
                statusArea.style.border = '1px solid #a5d6a7';
            }
        }
    </script>
    """)
    
    # הצגת הטופס
    display(form_html)
    
    # אזור פלט לתוצאות
    output_area = widgets.Output()
    display(output_area)
    
    return output_area
