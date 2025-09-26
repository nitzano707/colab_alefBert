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
from IPython.display import display, clear_output, Markdown, HTML
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

# מחלקת ניתוח (ללא שינוי)
class SeedSentenceAnalyzer:
    def __init__(self, seed_sentence, model="imvladikon/sentence-transformers-alephbert"):
        self.seed_sentence = seed_sentence.strip()
        self.model_name = model
        self.model = SentenceTransformer(model)
        self.sentences = []
        self.similarities = []
        self.df = None
        print(f"🌱 משפט הזרע: \"{self.seed_sentence}\"")
        print(f"🤖 מודל: {self.model_name}")

    # ... (שאר הפונקציות שלך נשארות ללא שינוי) ...





# טופס אינטראקטיבי (ipywidgets)
def create_analysis_form():
    clear_output()
    show_intro()

    display(Markdown("✅ **כדי להפעיל את הטופס יש ללחוץ על ▶️ (הפעלת תא) מצד שמאל**\n\n---"))

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
        description='🔍 הרצת ניתוח',
        button_style='success',
        layout=widgets.Layout(width='200px', height='40px')
    )

    def on_analyze_clicked(b):
        if not seed_text.value.strip():
            print("❌ יש להזין משפט זרע")
            return
        if not file_upload.value:
            print("❌ יש להעלות קובץ CSV")
            return

        uploaded_file = list(file_upload.value.values())[0]
        filename = 'uploaded_file.csv'
        with open(filename, 'wb') as f:
            f.write(uploaded_file['content'])

        print("🚀 מתחיל ניתוח...")
        analyzer = SeedSentenceAnalyzer(seed_text.value.strip())
        df = analyzer.load_sentences_from_csv(filename, column_name.value)
        if df is None:
            return
        similarities = analyzer.calculate_similarities_to_seed()
        if similarities is None:
            return
        analyzer.display_results(num_strong.value, num_medium.value)
        analyzer.create_visualizations()
        print("\n🎉 הניתוח הושלם בהצלחה!")

    analyze_button.on_click(on_analyze_clicked)

    form = widgets.VBox([
        seed_text, file_upload, column_name, num_strong, num_medium, analyze_button
    ])
    display(form)

