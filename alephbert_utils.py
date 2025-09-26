# -*- coding: utf-8 -*-
"""
alephbert_utils.py

פותח ע"י ד"ר ניצן אליקים | elyakim@talpiot.ac.il
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
plt.rcParams['font.family'] = 'DejaVu Sans'


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
            print("\u202B📂 קובץ CSV נטען עם", len(self.df), "שורות\u202C")

            if sentence_column not in self.df.columns:
                print("\u202B❌ עמודה '" + sentence_column + "' לא נמצאה\u202C")
                print("\u202B📋 עמודות זמינות:", list(self.df.columns), "\u202C")
                return None

            self.df = self.df.dropna(subset=[sentence_column])
            self.df = self.df[self.df[sentence_column].str.strip() != '']
            self.df = self.df[self.df[sentence_column].str.strip() != self.seed_sentence]
            self.df = self.df.reset_index(drop=True)

            self.sentences = self.df[sentence_column].tolist()
            print("\u202B✅ נטענו " + str(len(self.sentences)) + " משפטים תקינים\u202C")

            print("\u202B📝 דוגמה ממשפטים שנטענו:\u202C")
            for i, sentence in enumerate(self.sentences[:5], 1):
                print("\u202B" + f"{i:2d}. {sentence}" + "\u202C")
            if len(self.sentences) > 5:
                print("\u202B... ועוד " + str(len(self.sentences) - 5) + " משפטים\u202C")

            return self.df
        except Exception as e:
            print("\u202B❌ שגיאה בטעינת הקובץ:", e, "\u202C")
            return None

    def calculate_similarities_to_seed(self):
        if not self.sentences:
            print("\u202B❌ אין משפטים לעיבוד. טען קובץ CSV קודם.\u202C")
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
            print("\u202B❌ שגיאה בחישוב דמיון:", e, "\u202C")
            return None

    def display_results(self, num_strong=5, num_medium=5):
        all_matches = [(i, sentence, score) for i, (sentence, score) in enumerate(zip(self.sentences, self.similarities))]
        all_matches.sort(key=lambda x: x[2], reverse=True)

        strong = [m for m in all_matches if m[2] >= 0.75]
        medium = [m for m in all_matches if 0.70 <= m[2] < 0.75]
        weak = [m for m in all_matches if m[2] < 0.70]

        # כותרת מעוצבת בראש הניתוח
        display(HTML(f"""
        <div dir="rtl" align="right" style="border:2px solid #4CAF50; padding:10px; margin:10px 0; font-family:Arial;">
        <h3>📊 תוצאות ניתוח</h3>
        <b>פותח ע"י ד"ר ניצן אליקים | elyakim@talpiot.ac.il</b><br>
        🌱 משפט הזרע: "{self.seed_sentence}"<br>
        🤖 מודל: {self.model_name}
        </div>
        """))

        # פלט טקסטואלי
        if strong:
            show_n = min(num_strong, len(strong))
            print("\u202B🟢 דמיון חזק (≥0.75) | מציג " + str(show_n) + " מתוך " + str(len(strong)) + "\u202C")
            for rank, (i, s, sc) in enumerate(strong[:show_n], 1):
                print("\u202B" + f"#{rank:2d} | ציון: {sc:.4f}\n   📝 {s}" + "\u202C")

        if medium:
            show_n = min(num_medium, len(medium))
            print("\u202B🟡 דמיון בינוני (0.70–0.749) | מציג " + str(show_n) + " מתוך " + str(len(medium)) + "\u202C")
            for rank, (i, s, sc) in enumerate(medium[:show_n], 1):
                print("\u202B" + f"#{rank:2d} | ציון: {sc:.4f}\n   📝 {s}" + "\u202C")

        print("\u202B--- סיכום ---\u202C")
        print("\u202B🔥 חזק (≥0.75): " + str(len(strong)) + "\u202C")
        print("\u202B🟡 בינוני (0.70–0.749): " + str(len(medium)) + "\u202C")
        print("\u202B🔵 חלש (<0.70): " + str(len(weak)) + "\u202C")
        print("\u202Bסה\"כ: " + str(len(all_matches)) + " משפטים\u202C")
        if all_matches:
            best_score = all_matches[0][2]
            print("\u202B🎯 הציון הגבוה ביותר: " + f"{best_score:.4f}" + "\u202C")

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


# טופס אינטראקטיבי
def create_analysis_form():
    display(HTML("<div dir='rtl' align='right'><h3>📋 טופס ניתוח משפטי זרע</h3></div>"))
    print("\u202B📁 הוראות להכנת קובץ CSV:\n1. צור קובץ עם עמודה בשם 'sentence'\n2. כל שורה מכילה משפט אחד\n3. שמור את הקובץ ב-UTF-8\u202C")

    seed_text = widgets.Textarea(
        value='',
        placeholder='הכנס כאן את משפט הזרע...',
        description='משפט הזרע:',
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
        description='משפטים חזקים (≥0.75):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    num_medium = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description='משפטים בינוניים (0.70-0.749):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%')
    )

    analyze_button = widgets.Button(
        description='🔍 בצע ניתוח',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )

    output_area = widgets.Output()

    def on_analyze_clicked(b):
        with output_area:
            clear_output()
            print("\u202B🚀 מתחיל ניתוח...\u202C")
        b.disabled = True

        try:
            if not seed_text.value.strip():
                with output_area:
                    clear_output()
                    print("\u202B❌ יש להזין משפט זרע\u202C")
                return
            if not file_upload.value:
                with output_area:
                    clear_output()
                    print("\u202B❌ יש להעלות קובץ CSV\u202C")
                return

            uploaded_file = list(file_upload.value.values())[0]
            filename = 'uploaded_file.csv'
            with open(filename, 'wb') as f:
                f.write(uploaded_file['content'])

            analyzer = SeedSentenceAnalyzer(seed_text.value.strip())
            df = analyzer.load_sentences_from_csv(filename, column_name.value)
            if df is None:
                return
            similarities = analyzer.calculate_similarities_to_seed()
            if similarities is None:
                return

            with output_area:
                clear_output()  # מוחק את "🚀 מתחיל ניתוח..."
                analyzer.display_results(num_strong.value, num_medium.value)
                analyzer.create_visualizations()
                print("\u202B🎉 הניתוח הושלם בהצלחה!\u202C")

        finally:
            b.disabled = False  # הכפתור חוזר להיות פעיל

    analyze_button.on_click(on_analyze_clicked)

    rtl_box = widgets.VBox([
        seed_text, file_upload, column_name, num_strong, num_medium, analyze_button, output_area
    ], layout=widgets.Layout(direction='rtl', align_items='flex-end'))

    display(rtl_box)
