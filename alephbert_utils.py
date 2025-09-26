# -*- coding: utf-8 -*-
"""
alephbert_utils.py

פותח ע"י ד"ר ניצן אליקים
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import ipywidgets as widgets
from IPython.display import display, clear_output, HTML
from sentence_transformers import SentenceTransformer, util
import bidi.algorithm
from arabic_reshaper import reshape

warnings.filterwarnings('ignore')

# פונקציה לתיקון עברית (בשביל גרפים בלבד)
def fix_hebrew_text(text):
    try:
        reshaped_text = reshape(text)
        bidi_text = bidi.algorithm.get_display(reshaped_text)
        return bidi_text
    except:
        return text

# עיצוב גרפים
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")
plt.rcParams['font.family'] = 'Arial'

# מחלקת ניתוח
class SeedSentenceAnalyzer:
    def __init__(self, seed_sentence, model="imvladikon/sentence-transformers-alephbert"):
        self.seed_sentence = seed_sentence.strip()
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.sentences = []
        self.similarities = []
        self.df = None

    def load_sentences_from_csv(self, csv_path, sentence_column='sentence'):
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8')
            if sentence_column not in self.df.columns:
                print(f"❌ עמודה '{sentence_column}' לא נמצאה")
                print(f"📋 עמודות זמינות: {list(self.df.columns)}")
                return None

            self.df = self.df.dropna(subset=[sentence_column])
            self.df = self.df[self.df[sentence_column].str.strip() != '']
            self.df = self.df[self.df[sentence_column].str.strip() != self.seed_sentence]
            self.df = self.df.reset_index(drop=True)

            self.sentences = self.df[sentence_column].tolist()
            return self.df
        except Exception as e:
            print(f"❌ שגיאה בטעינת הקובץ: {e}")
            return None

    def calculate_similarities_to_seed(self):
        if not self.sentences:
            print("❌ אין היגדים לעיבוד. טען קובץ CSV קודם.")
            return None

        try:
            seed_emb = self.model.encode(self.seed_sentence, convert_to_tensor=True)
            sent_embs = self.model.encode(self.sentences, convert_to_tensor=True)
            self.similarities = util.cos_sim(seed_emb, sent_embs).cpu().numpy().flatten().tolist()

            if self.df is not None:
                self.df['similarity_score'] = self.similarities
                self.df = self.df.sort_values('similarity_score', ascending=False).reset_index(drop=True)

            return self.similarities
        except Exception as e:
            print(f"❌ שגיאה בחישוב דמיון: {e}")
            return None

    def display_results(self, num_strong=5, num_medium=5):
        if not self.similarities:
            print("❌ עדיין לא חושבו ציונים")
            return None

        all_matches = [(i, sentence, score) for i, (sentence, score) in enumerate(zip(self.sentences, self.similarities))]
        all_matches.sort(key=lambda x: x[2], reverse=True)

        strong = [m for m in all_matches if m[2] >= 0.75]
        medium = [m for m in all_matches if 0.70 <= m[2] < 0.75]
        weak = [m for m in all_matches if m[2] < 0.70]

        # כותרת ותצוגה מסודרת
        header_html = f"""
        <div dir="rtl" style="text-align:right; font-family:Arial; border:2px solid green; padding:10px; margin:10px 0;">
            <h3>📊 תוצאות ניתוח</h3>
            <p><b>פותח ע"י ד"ר ניצן אליקים | elyakim@talpiot.ac.il</b></p>
            <p>🌱 היגד זרע: "{self.seed_sentence}"</p>
            <p>🤖 מודל: {self.model_name}</p>
        </div>
        """
        display(HTML(header_html))

        # סיכום מילולי קצר
        summary_text = f"""
        <div dir="rtl" style="text-align:right; font-family:Arial; padding:5px;">
            <p>נמצאו <b>{len(strong)}</b> היגדים דומים מאוד (≥0.75), 
            <b>{len(medium)}</b> היגדים דומים במידה בינונית (0.70–0.749),
            ו־<b>{len(weak)}</b> היגדים רחוקים במשמעות.</p>
            <p>סה"כ נותחו <b>{len(all_matches)}</b> היגדים.</p>
        </div>
        """
        display(HTML(summary_text))

        # טבלה עם Top-N היגדים (חזקים ובינוניים בלבד)
        top_items = strong[:num_strong] + medium[:num_medium]
        if top_items:
            df_top = pd.DataFrame(
                [(s, f"{sc:.3f}") for (_, s, sc) in top_items],
                columns=["היגד", "ציון דמיון"]
            )
            display(df_top.style.set_table_styles(
                [{'selector': 'th', 'props': [('text-align', 'right'), ('font-family', 'Arial')]}]
            ).set_properties(**{'text-align': 'right', 'font-family': 'Arial'}))

        return strong, medium, weak, all_matches

    def create_visualizations(self):
        scores = np.array(self.similarities)
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))

        categories = [fix_hebrew_text('דמיון חזק (≥0.75)'),
                      fix_hebrew_text('דמיון בינוני (0.70-0.749)'),
                      fix_hebrew_text('דמיון חלש (<0.70)')]
        counts = [len([s for s in scores if s >= 0.75]),
                  len([s for s in scores if 0.70 <= s < 0.75]),
                  len([s for s in scores if s < 0.70])]
        axes[0,0].bar(categories, counts, color=['green','gold','skyblue'])
        axes[0,0].set_title(fix_hebrew_text("התפלגות לפי רמות דמיון"))

        axes[0,1].hist(scores, bins=20, alpha=0.7, color='lightblue', edgecolor='black')
        axes[0,1].axvline(np.mean(scores), color='red', linestyle='--', label=fix_hebrew_text(f'ממוצע: {np.mean(scores):.3f}'))
        axes[0,1].axvline(np.median(scores), color='green', linestyle='--', label=fix_hebrew_text(f'חציון: {np.median(scores):.3f}'))
        axes[0,1].legend()
        axes[0,1].set_title(fix_hebrew_text("התפלגות ציוני דמיון"))

        top_15_idx = np.argsort(scores)[-15:][::-1]
        top_15_scores = scores[top_15_idx]
        axes[0,2].bar(range(1,16), top_15_scores, color='green')
        axes[0,2].axhline(y=0.75, color='darkgreen', linestyle='--', label=fix_hebrew_text('חזק (0.75)'))
        axes[0,2].axhline(y=0.70, color='orange', linestyle='--', label=fix_hebrew_text('בינוני (0.70)'))
        axes[0,2].legend()
        axes[0,2].set_title(fix_hebrew_text("טופ 15 ציוני דמיון"))

        axes[1,0].boxplot(scores, patch_artist=True, boxprops=dict(facecolor='lightgreen', alpha=0.7))
        axes[1,0].axhline(y=0.75, color='darkgreen', linestyle='--')
        axes[1,0].axhline(y=0.70, color='orange', linestyle='--')
        axes[1,0].set_title(fix_hebrew_text("Box Plot"))

        top_20_idx = np.argsort(scores)[-20:][::-1]
        heatmap_data = scores[top_20_idx].reshape(4, 5)
        im = axes[1,1].imshow(heatmap_data, cmap='RdYlGn', vmin=0, vmax=1)
        for i in range(4):
            for j in range(5):
                axes[1,1].text(j, i, f"{heatmap_data[i,j]:.3f}", ha="center", va="center", color="black")
        axes[1,1].set_title(fix_hebrew_text("מפת חום - טופ 20"))
        fig.colorbar(im, ax=axes[1,1])

        axes[1,2].axis('off')
        plt.tight_layout(pad=3.0)
        plt.show()


# טופס אינטראקטיבי (ipywidgets)
def create_analysis_form():
    instructions = widgets.HTML("""
    <div dir="rtl" style="text-align:right; font-family:Arial;">
    <b>📁 הוראות להכנת קובץ CSV:</b><br>
    1. צור קובץ עם עמודה בשם 'sentence'<br>
    2. כל שורה מכילה היגד אחד<br>
    3. שמור את הקובץ ב־UTF-8
    </div>
    """)

    seed_text = widgets.Textarea(
        value='',
        placeholder='הכנס כאן את היגד הזרע...',
        description='היגד הזרע:',
        layout=widgets.Layout(width='80%')
    )

    file_upload = widgets.FileUpload(
        accept='.csv',
        multiple=False,
        description='צירוף קובץ'
    )

    column_name = widgets.Text(
        value='sentence',
        description='עמודת טקסט:',
        layout=widgets.Layout(width='50%')
    )

    num_strong = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description='היגדים חזקים (≥0.75):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    num_medium = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description='היגדים בינוניים (0.70-0.749):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    analyze_button = widgets.Button(
        description='🔍 בצע ניתוח',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )

    export_button = widgets.Button(
        description='⬇️ ייצוא לאקסל',
        button_style='info',
        layout=widgets.Layout(width='200px', height='40px')
    )
    export_button.disabled = True

    output_area = widgets.Output()

    def on_analyze_clicked(b):
        with output_area:
            clear_output()
            print("🚀 מתחיל ניתוח...")
        analyze_button.disabled = True

        if not seed_text.value.strip():
            with output_area:
                clear_output()
                print("❌ יש להזין היגד זרע")
            analyze_button.disabled = False
            return
        if not file_upload.value:
            with output_area:
                clear_output()
                print("❌ יש להעלות קובץ CSV")
            analyze_button.disabled = False
            return

        uploaded_file = list(file_upload.value.values())[0]
        filename = 'uploaded_file.csv'
        with open(filename, 'wb') as f:
            f.write(uploaded_file['content'])

        analyzer = SeedSentenceAnalyzer(seed_text.value.strip())
        df = analyzer.load_sentences_from_csv(filename, column_name.value)
        if df is None:
            analyze_button.disabled = False
            return
        similarities = analyzer.calculate_similarities_to_seed()
        if similarities is None:
            analyze_button.disabled = False
            return

        with output_area:
            clear_output()
            strong, medium, weak, all_matches = analyzer.display_results(num_strong.value, num_medium.value)
            analyzer.create_visualizations()
            print("\n🎉 הניתוח הושלם בהצלחה!")

        # הפעלת כפתור ייצוא
        def export_to_excel(btn):
            df_out = pd.DataFrame([(s, sc) for (_, s, sc) in all_matches],
                                  columns=["היגד", "ציון דמיון"])
            df_out.to_excel("תוצאות_ניתוח.xlsx", index=False)
            with output_area:
                print("📂 קובץ 'תוצאות_ניתוח.xlsx' נוצר בתיקייה הנוכחית.")

        export_button.on_click(export_to_excel)
        export_button.disabled = False
        analyze_button.disabled = False

    analyze_button.on_click(on_analyze_clicked)

    form = widgets.VBox([
        instructions, seed_text, file_upload, column_name,
        num_strong, num_medium, analyze_button, export_button, output_area
    ])
    display(form)
