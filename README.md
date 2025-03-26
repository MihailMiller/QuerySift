# QuerySift: Multi-Document Question Answering

## Overview

If you have a collection of documents on a specific topic and a set of unanswered questions, this tool helps you quickly skim through the material, extract key information, and provide an entry point for deeper research.

### What It Can Do:

- ✅ **Provide an overview of the material** (useful for learning, research, interviews, etc.)
- ✅ **Extract answers** from multiple documents based on predefined questions
- ✅ **Highlight key information** in text files automatically
- ✅ **Generate structured reports** in Excel and Word for easy review

### What It Cannot Do:

- ❌ **Replace human judgment** – This tool is purely extractive, meaning it retrieves relevant excerpts but does not create new content or provide deep contextual understanding. Final review and interpretation are up to the user.

---

## How It Works

1️⃣ **Loads Questions:** Reads predefined questions from a JSON file.

2️⃣ **Splits Large Texts:** Breaks down big text files into smaller pieces.

3️⃣ **Finds Relevant Matches:** Uses AI to compare text pieces with the questions.

4️⃣ **Highlights Important Text:** Automatically marks key phrases.

5️⃣ **Removes Duplicates:** Keeps only the best and most relevant matches.

6️⃣ **Saves to Excel & Word:** Organizes results neatly for easy access.

---

## What You Need Before Running

1️⃣ **Text Files:** Put them in a folder called `texts/`

2️⃣ **Predefined Questions:** Ensure `category_questions.json` is a structured JSON file containing predefined questions categorized by topics of interest. It should be formatted as follows:

```json
{
  "Technology": {
    "Artificial Intelligence": {
      "Deep Learning": ["What are the applications of AI?", "How does deep learning improve AI?"],
      "Machine Learning": ["What are the different types of machine learning?", "How does supervised learning work?"]
    },
    "Natural Language Processing": {
      "Text Analysis": ["What are the key challenges in NLP?", "How does sentiment analysis work?"],
      "Speech Recognition": ["How do voice assistants work?", "What is the role of transformers in NLP?"]
    },
    "Computer Vision": {
      "Image Processing": ["What are the applications of computer vision?", "How do convolutional neural networks (CNNs) work?"],
      "Object Detection": ["How does YOLO work?", "What are the latest advancements in object detection?"]
    }
  }
}
```

---

## 🛠️ Installation & Setup

1️⃣ Install the required Python libraries by running:

```bash
pip install pandas numpy torch transformers scikit-learn tqdm python-docx openpyxl einops
```

2️⃣ Run the script using:

```bash
python querysift.py
```

You can also customize the settings using command-line arguments (see below 👇).

---

## ⚙️ Customization Options (Command-Line Arguments)

You can tweak how the script works by using these options:

| Option                    | Default Value                       | What It Does                                |
| ------------------------- | ----------------------------------- | ------------------------------------------- |
| `--chunk_size`            | `1000`                              | How big each text chunk should be           |
| `--chunk_overlap`         | `500`                               | How much overlap each chunk should have     |
| `--highlight_limit`       | `5`                                 | How many key phrases should be highlighted? |
| `--top_results_limit`     | `50`                                | How many excerpts should be retrieved?      |
| `--minimum_merge_overlap` | `10`                                | Minimum words required to merge chunks      |
| `--embedding_model`       | `jinaai/jina-embeddings-v3`         | AI model used to find similar texts         |
| `--qa_model`              | `deepset/gelectra-large-germanquad` | AI model used for key phrases               |

---

## 📁 Folder & File Structure

```
querysift/
│── texts/                     # Folder with text files to analyze
│── category_questions.json    # List of predefined questions
│── output_results.xlsx        # Excel file with the best results
│── analysis_word/             # Word reports with highlighted phrases
│── querysift.py               # The main Python script
```

---

## What You Get

📊 `output_results.xlsx` → An Excel file with best matches, sorted by relevance

📄 `analysis_word/` → Word documents with highlighted key phrases for each query

---

## License

Licensed under the **MIT License** – free to use and modify!

---

## Funding

Parts of the work were funded by grants of the German Ministry of Education and Research in the context of the joint research project "MANGAN" (01IS22011C) under the supervision of the PT-DLR.
