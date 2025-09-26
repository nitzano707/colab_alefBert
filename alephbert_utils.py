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
from IPython.display import display, clear_output
from sentence_transformers import SentenceTransformer, util
import bidi.algorithm
from arabic_reshaper import reshape
from google.colab import files

# התקנת Arial בסביבת Colab
!apt-get -y install ttf-mscorefonts-installer fontconfig &> /dev/null
!fc-cache -fv &> /dev/null

# הגדרות גרפים ועיצוב
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Arial'
sns.set_style("whitegrid")

# פונקציה לתיקון עברית (גרפים בלבד)
def fix_hebrew_text(text):
    try:
        reshaped_text = reshape(text)
        bidi_text = bidi.algorithm.get_display(reshaped_text)
        return bidi_text
    except:
        return text

# מחלקת ניתוח
class SeedSentenceAnalyzer:
    def __init__(self, seed_sentence, model="imvladikon/sentence-transformers-alephbert"):
        self.seed_sentence = seed_sentence.strip()
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.sentences = []
        self.similarities = []
        self.df = None

        print("==============================================")
        print("📊 ניתוח היגדים באמצעות AlephBERT")
        print("פותח ע\"י: ד\"ר ניצן אליקים | elyakim@talpiot.ac.il")
        print("==============================================")
        print(f"🌱 משפט הזרע: \"{self.seed_sentence}\"")
        print(f"🤖 מודל: {self.model_name}\n")

    def load_sentences_from_csv(self, csv_path, sentence_column='sentence'):
        try:
            self.df = pd.read_csv(csv_path, encoding='utf-8')
            if sentence_column not in self.df.columns:
                print(f"❌ עמודה '{sentence_column}' לא נמצאה בקובץ")
                print(f"📋 עמודות זמינות: {list(self.df.columns)}")
                return None

            self.df = self.df.dropna(subset=[sentence_column])
            self.df = self.df[self.df[sentence_column].str.strip() != '']
            self.df = self.df[self.df[sentence_column].str.strip() != self.seed_sentence]
            self.df = self.df.reset_index(drop=True)

            self.sentences = self.df[sentence_column].tolist()
            print(f"📂 נטענו {len(self.sentences)} היגדים מתוך הקובץ")

            return self.df
        except Exception as e:
            print(f"❌ שגיאה בטעינת הקובץ: {e}")
            return None

    def calculate_similarities_to_seed(self):
        if not self.sentences:
            print("❌ אין היגדים לעיבוד. טען קובץ CSV קודם.")
            return None

        print(f"🔄 מחשב דמיון עבור {len(self.sentences)} היגדים למשפט הזרע...")
        print("⏳ זה הרבה יותר מהיר מאשר חישוב מטריצה מלאה!")

        try:
            seed_emb = self.model.encode(self.seed_sentence, convert_to_tensor=True)
            sent_embs = self.model.encode(self.sentences, convert_to_tensor=True)
            self.similarities = util.cos_sim(seed_emb, sent_embs).cpu().numpy().flatten().tolist()

            print(f"✅ הסתיים! חושבו {len(self.similarities)} השוואות")

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

        print("\n📑 תוצאות הניתוח:")
        print("=" * 100)

        # חזק
        if strong:
            show_n = min(num_strong, len(strong))
            print(f"\n🟢 דמיון חזק (≥0.75) ({len(strong)} היגדים): מציג {show_n}")
            print("-" * 100)
            for rank, (i, s, sc) in enumerate(strong[:show_n], 1):
                print(f"🟢 #{rank:2d} | ציון: {sc:.4f} ({sc*100:.1f}%) | היגד {i+1:3d}")
                print(f"    \"{s}\"")
                print()

        # בינוני
        if medium:
            show_n = min(num_medium, len(medium))
            print(f"\n🟡 דמיון בינוני (0.70-0.749) ({len(medium)} היגדים): מציג {show_n}")
            print("-" * 100)
            for rank, (i, s, sc) in enumerate(medium[:show_n], 1):
                print(f"🟡 #{rank:2d} | ציון: {sc:.4f} ({sc*100:.1f}%) | היגד {i+1:3d}")
                print(f"    \"{s}\"")
                print()

        print(f"\n📊 סיכום:")
        print(f"🔥 חזק (≥0.75): {len(strong):3d} היגדים")
        print(f"🟡 בינוני (0.70-0.749): {len(medium):3d} היגדים")
        print(f"🔵 חלש (<0.70): {len(weak):3d} היגדים")
        print(f"📝 סה\"כ: {len(all_matches):3d} היגדים")
        if all_matches:
            best_score = all_matches[0][2]
            print(f"🎯 הציון הגבוה ביותר: {best_score:.4f} ({best_score*100:.1f}%)")

        print("\n📝 סיכום מילולי:")
        print("מרבית ההיגדים שנבחנו נמצאים ברמת דמיון גבוהה או בינונית. מומלץ לבחור מתוכם את ההיגדים החזקים, "
              "ולשקול מחדש את אלה שציוניהם חלשים יותר.")

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

    def export_to_excel(self, colab_link="https://colab.research.google.com/"):
        if self.df is None:
            print("❌ אין נתונים לייצוא")
            return

        filename = "results.xlsx"
        with pd.ExcelWriter(filename, engine='xlsxwriter') as writer:
            workbook = writer.book
            worksheet = workbook.add_worksheet("Results")
            writer.sheets["Results"] = worksheet

            bold = workbook.add_format({'bold': True})

            worksheet.write("A1", "משפט הזרע", bold)
            worksheet.write("B1", self.seed_sentence, bold)

            worksheet.write("A2", f"הקובץ נוצר ע\"י מחברת COLAB: {colab_link}")

            self.df.to_excel(writer, sheet_name="Results", startrow=4, index=False)

        print(f"💾 הקובץ נשמר בשם {filename}")
        files.download(filename)

# טופס אינטראקטיבי
def create_analysis_form():
    seed_text = widgets.Textarea(
        value='',
        placeholder='הכנס כאן את משפט הזרע...',
        description='משפט הזרע:',
        layout=widgets.Layout(width='80%', direction='rtl')
    )

    file_upload = widgets.FileUpload(
        accept='.csv',
        multiple=False,
        description='צירוף קובץ',
        layout=widgets.Layout(direction='rtl')
    )

    column_name = widgets.Text(
        value='sentence',
        description='עמודת טקסט:',
        layout=widgets.Layout(width='50%', direction='rtl')
    )

    num_strong = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description='היגדים חזקים (≥0.75):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', direction='rtl')
    )

    num_medium = widgets.IntSlider(
        value=5, min=0, max=50, step=1,
        description='היגדים בינוניים (0.70-0.749):',
        style={'description_width': 'initial'},
        layout=widgets.Layout(width='80%', direction='rtl')
    )

    analyze_button = widgets.Button(
        description='🔍 בצע ניתוח',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )

    export_button = widgets.Button(
        description='⬇️ הורדת אקסל',
        button_style='info',
        layout=widgets.Layout(width='200px', height='40px')
    )
    export_button.disabled = True

    output_area = widgets.Output(layout={'border': '1px solid gray'})

    def on_analyze_clicked(b):
        with output_area:
            clear_output()
            print("🚀 מתחיל ניתוח...")
        analyze_button.disabled = True

        if not seed_text.value.strip():
            with output_area:
                clear_output()
                print("❌ יש להזין משפט זרע")
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
            analyzer.display_results(num_strong.value, num_medium.value)
            analyzer.create_visualizations()

        export_button.on_click(lambda b: analyzer.export_to_excel())
        export_button.disabled = False
        analyze_button.disabled = False

    analyze_button.on_click(on_analyze_clicked)

    form = widgets.VBox([
        seed_text, file_upload, column_name, num_strong, num_medium,
        widgets.HBox([analyze_button, export_button]),
        output_area
    ], layout=widgets.Layout(direction='rtl'))

    display(form)
