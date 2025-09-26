# -*- coding: utf-8 -*-
"""
alephbert_utils.py
נבנה מתוך המחברת "יצירת היגדים לשאלון.ipynb"
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


# הצגת הקדמה (כפי שהיה במחברת)
def show_intro():
    intro_html = """
<div dir="rtl" align="right">

⚡<font color=green> <b><i>פותח ע"י ד"ר ניצן אליקים. <BR>
סרטון הסבר: <a href="https://youtu.be/tv9QXLJe2JU?feature=shared">https://youtu.be/tv9QXLJe2JU?feature=shared</a></b></i></font>
<hr>

# 📘 תקציר + קישורים

- **מקור השיטה:**  
  Hernandez & Nie (2023) מציגים מתודולוגיה לפיתוח פריטי שאלון באמצעות מודלים גנרטיביים ולאחר מכן סינון סמנטי בעזרת embeddings ו-cosine similarity, בתוספת ביקורת מומחים ואימות פסיכומטרי.  

- **יישום עדכני של השיטה:**  
  Liu et al. (2025) מפתחים סולם *אוריינות בינה מלאכותית יוצרת (GenAI)* בשיטה דומה: יצירת פריטים בעזרת AI, סינון סמנטי, ולבסוף EFA/CFA.     

- **מודל embeddings בשימוש במחברת זו:**  
  *AlephBERT* - גרסה עברית של  *BERT* המותאמת לניתוח סמנטי  
  🔗 https://huggingface.co/imvladikon/sentence-transformers-alephbert  

- חלופה מומלצת לרב־לשוניות: LaBSE (Google)  
  🔗 https://tfhub.dev/google/LaBSE/2  

---

# 🛠️ הסבר תמציתי של שלבי העבודה (יישום בשאלון)

## 1. הגדרת מבנה ו-Seed
- הגדר/י את הממדים התיאורטיים של המשתנה.  
- נסח/י משפט מייצג (seed) קצר וברור לכל ממד.  
**→ את המשפט הזה יש להקליד בטופס למטה במקום המתאים**

## 2. הפקה גנרטיבית של פריטים
- הפק/י מגוון רחב של וריאציות פריטים לכל seed באמצעות מודל גנרטיבי (LLM) כמו קלוד או ג'י.פי.טי.  
- מטרה: כיסוי רחב של המשמעות, לא נוסח "מושלם" בשלב זה.

## 3. המרה ל-Embeddings
- המר/י כל פריט (וגם את ה-seed) ל-sentence embeddings בעזרת מודל מתאים (AlephBERT במחברת זו).

## 4. סינון סמנטי אוטומטי ← **זה מה שהמחברת הזו עושה!**
- חשב/י cosine similarity בין כל פריט ל-seed שלו.  
- קבע/י סף (≈ 0.70–0.75 מקובל) לשמירת פריטים קרובים במשמעות.  
- הרחבות ביניים נשקלות תיאורטית; רחוקים נפסלים.

## 5. ביקורת מומחים ו"שיוף" ניסוחי
- מומחי תחום מסננים כפילויות, מתקנים ניסוחים, בודקים התאמה תרבותית/אתית.  
- שמירה על כיסוי מושגי מאוזן לכל תת-ממד.

## 6. אימות פסיכומטרי
- אסוף/י נתונים אמפיריים.  
- הרץ/י **EFA/CFA** ומהימנות פנימית (**Cronbach's α**).  
- אשר/י את מבנה הממדים ואת איכות הפריטים.  
- במידת הצורך: בצע/י קיצור סולם על בסיס עומסי גורם ותפקוד פריט.

---

# 📚 מקורות מרכזיים
</div>

<div dir="ltr" align="left">

Hernandez, I., & Nie, W. (2023). The AI-IP: Minimizing the guesswork of personality scale item development through artificial intelligence. Personnel Psychology, 76(4), 1011–1035. 📄 DOI: https://doi.org/10.1111/peps.12543  

Liu, X., Zhang, L., & Wei, X. (2025). Generative Artificial Intelligence Literacy: Scale Development and Its Effect on Job Performance. Behavioral Sciences, 15(6), 811. 🌐 קישור פתוח: https://doi.org/10.3390/bs15060811  

- מודל AlephBERT: https://huggingface.co/imvladikon/sentence-transformers-alephbert  
- חלופה: LaBSE (Google) https://tfhub.dev/google/LaBSE/2  

</div>
"""
    display(HTML(intro_html))


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
