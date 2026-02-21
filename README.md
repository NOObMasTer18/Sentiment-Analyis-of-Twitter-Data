# Sentiment Analysis of Twitter Data

Train and evaluate a sentiment classifier on a Twitter dataset, then visualize sentiment patterns across entities and generate word clouds.

This project loads the provided training and validation CSVs, cleans and preprocesses tweet text, trains a **TF–IDF + Multinomial Naive Bayes** model, prints an evaluation dashboard (accuracy/precision/recall/F1 + classification report), and displays visualizations (confusion matrix, sentiment distributions, word clouds, and top indicative words per class).

## Repository Structure

- `src/main.py` — end-to-end pipeline (load → clean → preprocess → train → evaluate → visualize)
- `data/`
  - `twitter_training.csv`
  - `twitter_validation.csv`
- `Requirements.txt` — Python dependencies

## Requirements

- Python 3.9+ recommended

Install dependencies:

```bash
pip install -r Requirements.txt
```

## Data

The script expects these files to exist:

- `data/twitter_training.csv`
- `data/twitter_validation.csv`

Columns are renamed inside the script to:

- `TweetID`, `Entity`, `Sentiment`, `Tweet_Content`

Only the sentiment classes **Positive**, **Negative**, and **Neutral** are used.

## How to Run

From the repository root:

```bash
python src/main.py
```

What you’ll see:

- Dataset shape before/after cleaning
- Class distribution
- Model metrics (accuracy/precision/recall/F1) + full classification report
- Plots:
  - confusion matrix heatmap
  - overall sentiment distribution
  - entity-wise sentiment distribution
  - word clouds for each sentiment
  - top words per sentiment class (from Naive Bayes log probabilities)

## Method Summary

1. **Cleaning**: drop NA rows and duplicates
2. **Preprocessing** (`clean_text`): lowercase, remove URLs/mentions/hashtags, keep alphabetic characters, normalize whitespace
3. **Vectorization**: TF–IDF with English stopwords (`max_features=5000`)
4. **Model**: `MultinomialNB`
5. **Evaluation**: standard classification metrics + confusion matrix
6. **Visualization**: sentiment plots + word clouds + top words per class

## Notes

- `src/main.py` uses `matplotlib` interactive windows (`plt.show()`). If you run this in a headless environment (like some servers), you may need to switch to a non-interactive backend.
- If you want reproducible results, the split is controlled by `random_state=42`.

## Author

- Name: Akshay Kumar Singh
- GitHub: [@NOObMasTer18](https://github.com/NOObMasTer18)
