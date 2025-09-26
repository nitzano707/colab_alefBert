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
    מציג טופס ניתוח היגדים. colab_notebook_link – ייכתב בשורת ההסבר באקסל.
    """
    # הוראות קצרות
    instructions = HTML(
        f"""
        <div dir="rtl" style="text-align:right;font-family:{_font_family};margin-bottom:8px;">
          <b>הוראות להכנת קובץ CSV</b><br>
          1. צור/י קובץ עם עמודה בשם <code>sentence</code>.<br>
          2. כל שורה מכילה היגד אחד.<br>
          3. שמור/שמרי את הקובץ בקידוד UTF-8.
        </div>
        """
    )

    seed_text = widgets.Textarea(
        value="",
        placeholder="הכנס/י כאן את היגד הזרע…",
        description="היגד הזרע:",
        layout=widgets.Layout(width="80%"),
        style={'description_width': 'initial'}
    )

    file_upload = widgets.FileUpload(
        accept=".csv",
        multiple=False,
        description="צירוף קובץ",
        style={'description_width': 'initial'}
    )

    column_name = widgets.Text(
        value="sentence",
        description="עמודת טקסט:",
        layout=widgets.Layout(width="40%"),
        style={'description_width': 'initial'}
    )

    num_strong = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description="היגדים חזקים (≥0.75):",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width="80%")
    )

    num_medium = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description="היגדים בינוניים (0.70–0.749):",
        style={'description_width': 'initial'},
        layout=widgets.Layout(width="80%")
    )

    analyze_button = widgets.Button(
        description="בצע ניתוח",
        button_style="success",
        layout=widgets.Layout(width="200px", height="40px")
    )

    export_button = widgets.Button(
        description="ייצוא לאקסל",
        button_style="info",
        layout=widgets.Layout(width="200px", height="40px"),
        disabled=True
    )

    output_area = widgets.Output()  # כאן יוצגו התוצאות
    status_area = widgets.Output()  # הודעת "מתחיל ניתוח…"

    # משתנה שיחזיק את האנלייזר האחרון לייצוא אקסל
    current_analyzer: Optional[SeedSentenceAnalyzer] = None

    def run_analysis(_):
        nonlocal current_analyzer

        # ניקוי פלט ישן והצגת הודעת מצב מיידית
        with output_area:
            clear_output()

        with status_area:
            clear_output()
            print("מתחיל ניתוח...")     # מוצג מיידית
        analyze_button.disabled = True
        export_button.disabled = True

        # השהיה קצרה כדי להרנדר את ההודעה לפני עיבוד כבד
        time.sleep(0.05)

        # ולידציות
        if not seed_text.value.strip():
            with status_area:
                clear_output()
            with output_area:
                print("❌ יש להזין היגד זרע.")
            analyze_button.disabled = False
            return
        if not file_upload.value:
            with status_area:
                clear_output()
            with output_area:
                print("❌ יש להעלות קובץ CSV.")
            analyze_button.disabled = False
            return

        # שמירת הקובץ שהועלה
        uploaded_file = list(file_upload.value.values())[0]
        csv_path = "uploaded_file.csv"
        with open(csv_path, "wb") as f:
            f.write(uploaded_file["content"])

        # הרצה
        analyzer = SeedSentenceAnalyzer(seed_text.value.strip())
        df = analyzer.load_sentences_from_csv(csv_path, column_name.value)
        if df is None:
            with status_area:
                clear_output()
            analyze_button.disabled = False
            return

        sims = analyzer.calculate_similarities_to_seed()
        if sims is None:
            with status_area:
                clear_output()
            analyze_button.disabled = False
            return

        # הצגת תוצאות – ראשית ננקה את הודעת הסטטוס
        with status_area:
            clear_output()

        with output_area:
            clear_output()
            analyzer.show_header()
            analyzer.display_results(num_strong.value, num_medium.value)
            analyzer.create_visualizations()
            display(HTML(f'<div dir="rtl" style="text-align:right;font-family:{_font_family};margin-top:6px;">הניתוח הושלם.</div>'))

        # שמירת האנלייזר לייצוא
        current_analyzer = analyzer
        export_button.disabled = False
        analyze_button.disabled = False

    def export_excel(_):
        if current_analyzer is None:
            return
        with output_area:
            print("מייצא לאקסל...")
        current_analyzer.export_to_excel(colab_link=colab_notebook_link)

    # חיבור אירועים (handler יחיד, לא נערום מאזינים)
    analyze_button.on_click(run_analysis)
    export_button.on_click(export_excel)

    # סידור RTL לכל הקומפוננטות
    container = widgets.VBox(
        [
            instructions,
            seed_text,
            file_upload,
            column_name,
            num_strong,
            num_medium,
            widgets.HBox([analyze_button, export_button], layout=widgets.Layout(justify_content="flex-start")),
            status_area,
            output_area,
        ],
        layout=widgets.Layout(direction="rtl")  # RTL כולל לכל הצאצאים
    )
    display(container)
