# SENTIMENT ANALYSIS — QATAR AIRWAYS REVIEWS
#IMPORT LIBRARIES

import pandas as pd
import numpy as np
import re, string
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression, LinearRegression

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": "--",
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
})
COLORS = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
OUT = r"D:\22DH711684\FINAL PROJECT" + "\\"

# ── 1. LOAD DATA ──────────────────────────────────────────────
df = pd.read_csv(r"D:\22DH711684\FINAL PROJECT\qatar_airways_reviews_FP.csv")
data = df[["Review Body", "Rating", "Seat Type"]].dropna().copy()
print(f"Dataset: {len(data)} reviews")

# ── 2. LABEL ──────────────────────────────────────────────────
def convert_sentiment(s):
    return "positive" if s >= 7 else "neutral" if s >= 4 else "negative"

data["sentiment"] = data["Rating"].apply(convert_sentiment)
counts = data["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
print(counts)

# ── 3. STOP WORDS ─────────────────────────────────────────────
AIRLINE_STOP = {
    "flight","qatar","airways","airline","doha","seat","seats",
    "service","staff","crew","plane","aircraft","airport","class",
    "food","time","passenger","boarding","cabin","business","economy",
    "first","fly","flew","flying","ticket","gate","hour","hours",
    "day","days","travel","trip","luggage","bag","baggage","like",
    "got","get","also","would","said","told","one","two","three",
    "way","back","came","come","going","went","take","taken","make",
    "review","verified","new","really","passengers","flights","left",
    "right","next","well","even","still","just","much","many","every",
}
ALL_STOP = set(ENGLISH_STOP_WORDS) | AIRLINE_STOP

# ── 4. PREPROCESS (cho ML) ────────────────────────────────────
def preprocess(t):
    t = str(t).lower()
    t = re.sub(r"\d+", "", t)
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w not in ALL_STOP and len(w) > 2]
    return " ".join(tokens)

data["clean_text"] = data["Review Body"].apply(preprocess)
data["text_length"] = data["Review Body"].apply(lambda x: len(str(x).split()))

# ── 5. PREPROCESS CHI LAY TINH TU (cho WordCloud) ─────────────
ADJ_WORDS = {
    # Positive
    "excellent","good","great","amazing","wonderful","fantastic","superb",
    "comfortable","friendly","clean","helpful","polite","professional",
    "attentive","outstanding","smooth","spacious","delicious","enjoyable",
    "pleasant","satisfied","happy","impressed","brilliant","nice","perfect",
    "luxurious","efficient","lovely","beautiful","incredible","remarkable",
    "warm","generous","relaxing","refreshing","tasty","cozy","attentive",
    "responsive","punctual","organized","welcoming","courteous","charming",
    # Neutral
    "average","okay","decent","acceptable","reasonable","standard","normal",
    "adequate","ordinary","mixed","lukewarm","mediocre","moderate","fair",
    "typical","usual","basic","simple","plain","regular","limited",
    # Negative
    "bad","terrible","awful","horrible","disgusting","poor","worst","rude",
    "dirty","slow","cold","uncomfortable","cramped","broken","late","lost",
    "disappointing","unprofessional","unfriendly","unhelpful","arrogant",
    "chaotic","wrong","missing","stale","delayed","cancelled","overpriced",
    "crowded","noisy","smelly","faulty","damaged","outdated","unpleasant",
    "frustrating","annoying","exhausting","shocking","unacceptable","appalling",
}

def preprocess_adj(t):
    t = str(t).lower()
    t = t.translate(str.maketrans("", "", string.punctuation))
    tokens = [w for w in t.split() if w in ADJ_WORDS]
    return " ".join(tokens)

data["adj_text"] = data["Review Body"].apply(preprocess_adj)

# ── 6. LEXICON SCORE ──────────────────────────────────────────
POS_LEXICON = {
    "good":1,"great":2,"excellent":3,"amazing":2,"love":2,"perfect":3,
    "comfortable":1,"friendly":1,"outstanding":2,"wonderful":2,"best":2,
    "fantastic":2,"superb":2,"brilliant":2,"nice":1,"clean":1,"smooth":1,
    "professional":1,"helpful":1,"polite":1,"efficient":1,"delicious":1,
    "spacious":1,"enjoyable":1,"pleasant":1,"recommend":1,"impressed":2,
    "satisfied":1,"happy":1,"luxurious":1,"attentive":1,
}
NEG_LEXICON = {
    "bad":-1,"terrible":-3,"poor":-2,"hate":-2,"awful":-3,"worst":-3,
    "delay":-2,"dirty":-2,"rude":-2,"horrible":-3,"disgusting":-3,
    "disappointing":-2,"unprofessional":-1,"unfriendly":-1,"uncomfortable":-2,
    "cramped":-1,"broken":-2,"cancelled":-3,"refused":-2,"complaint":-1,
    "slow":-1,"cold":-1,"stale":-1,"late":-2,"lost":-2,"unhelpful":-2,
    "arrogant":-2,"chaotic":-2,"wrong":-1,"error":-1,"missing":-1,
}

def lexicon_score(text):
    words = str(text).lower().split()
    score = sum(POS_LEXICON.get(w, 0) + NEG_LEXICON.get(w, 0) for w in words)
    return score / max(len(words), 1)

def lexicon_sent(t):
    s = lexicon_score(t)
    return "positive" if s > 0.02 else "negative" if s < -0.02 else "neutral"

data["vader_score"] = data["Review Body"].apply(lexicon_score)
vader_pred = data["Review Body"].apply(lexicon_sent)
vader_acc  = accuracy_score(data["sentiment"], vader_pred)

# ── 7. RULE BASED ─────────────────────────────────────────────
def rule_sent(text):
    pos = sum(1 for w in text.split() if w in POS_LEXICON)
    neg = sum(1 for w in text.split() if w in NEG_LEXICON)
    return "positive" if pos > neg else "negative" if neg > pos else "neutral"

rule_pred = data["clean_text"].apply(rule_sent)
rule_acc  = accuracy_score(data["sentiment"], rule_pred)

# ── 8. TF-IDF + ML ────────────────────────────────────────────
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(data["clean_text"])
y = data["sentiment"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

models = {
    "Naive Bayes":         MultinomialNB(),
    "SVM":                 LinearSVC(max_iter=2000),
    "Logistic Regression": LogisticRegression(max_iter=300),
}
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    acc  = accuracy_score(y_test, pred)
    results[name] = acc
    print(f"\n{name} — Accuracy: {acc:.4f}")
    print(classification_report(y_test, pred))

results["Rule-Based"]           = rule_acc
results["Lexicon (VADER-like)"] = vader_acc

# ============================================================
# BIEU DO 1 — SENTIMENT DISTRIBUTION
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(counts.index, counts.values,
              color=[COLORS[s] for s in counts.index],
              width=0.5, edgecolor="white", linewidth=2)
for bar, val in zip(bars, counts.values):
    pct = val / len(data) * 100
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 8,
            f"{val}\n({pct:.1f}%)", ha="center", fontsize=11, fontweight="bold")
ax.set_title("Sentiment Distribution — Qatar Airways Reviews")
ax.set_xlabel("Sentiment"); ax.set_ylabel("Number of Reviews")
ax.set_ylim(0, counts.max() * 1.22)
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "01_sentiment_distribution.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BIEU DO 2 — TEXT LENGTH DISTRIBUTION
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
for label in ["positive", "neutral", "negative"]:
    subset = data[data["sentiment"] == label]["text_length"]
    ax.hist(subset, bins=40, alpha=0.6,
            label=f"{label.capitalize()} (mean={subset.mean():.0f} words)",
            color=COLORS[label], edgecolor="white")
ax.set_title("Review Text Length Distribution by Sentiment")
ax.set_xlabel("Number of Words"); ax.set_ylabel("Frequency")
ax.legend(title="Sentiment")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "02_text_length_distribution.png", dpi=150, bbox_inches="tight")
plt.show()


# ============================================================
# BIEU DO 3 — WORDCLOUD (tinh tu dac trung)
# ============================================================
def get_adj_words(target_label, data, min_count=3, ratio=1.3):
    if target_label == "neutral":
        # Neutral dùng tất cả từ, không chỉ tính từ
        target = Counter(" ".join(data[data["sentiment"]=="neutral"]["clean_text"]).split())
    else:
        target = Counter(" ".join(data[data["sentiment"]==target_label]["adj_text"]).split())
    other = Counter(" ".join(data[data["sentiment"]!=target_label]["clean_text"]).split())
    unique = {w: c for w, c in target.items()
              if c / other.get(w, 1) > ratio and c >= min_count}
    return unique

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
configs = [
    ("positive", "Greens"),
    ("neutral",  "Oranges"),
    ("negative", "Reds"),
]
for ax, (label, cmap) in zip(axes, configs):
    wf = get_adj_words(label, data)
    if len(wf) < 5:
        # neutral thực sự không có từ đặc trưng → hiển thị thông báo
        ax.text(0.5, 0.5, "Not enough\ndistinctive words\nfor Neutral",
                ha="center", va="center", fontsize=14,
                color=COLORS[label], transform=ax.transAxes)
        ax.axis("off")
    else:
        wc = WordCloud(width=600, height=350, background_color="white",
                       colormap=cmap, max_words=80,
                       collocations=False).generate_from_frequencies(wf)
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
    ax.set_title(f"WordCloud — {label.capitalize()}", fontsize=13,
                 fontweight="bold", color=COLORS[label])
fig.suptitle("WordCloud by Sentiment — Qatar Airways", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.savefig(OUT + "03_wordcloud_by_sentiment.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved 03_wordcloud_by_sentiment.png")

# ============================================================
# BIEU DO 4 — TOP TF-IDF KEYWORDS
# ============================================================
feature_names = vectorizer.get_feature_names_out()
tfidf_scores  = X.mean(axis=0).A1
top_indices   = tfidf_scores.argsort()[-15:][::-1]
top_words     = [feature_names[i] for i in top_indices]
top_scores    = tfidf_scores[top_indices]

fig, ax = plt.subplots(figsize=(9, 6))
colors_bar = plt.cm.plasma(np.linspace(0.2, 0.8, 15))
ax.barh(range(15), top_scores[::-1], color=colors_bar, edgecolor="white", height=0.7)
ax.set_yticks(range(15))
ax.set_yticklabels(top_words[::-1], fontsize=11)
for i, score in enumerate(top_scores[::-1]):
    ax.text(score + 0.0003, i, f"{score:.4f}", va="center", fontsize=9)
ax.set_title("Top 15 Most Important Words (TF-IDF)")
ax.set_xlabel("Average TF-IDF Score")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "04_top_tfidf_keywords.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BIEU DO 5 — SENTIMENT BY SEAT TYPE
# ============================================================
seat_order = [s for s in ["Economy Class","Business Class","First Class","Premium Economy"]
              if s in data["Seat Type"].unique()]
seat_sentiment = pd.crosstab(data["Seat Type"], data["sentiment"])
seat_sentiment = seat_sentiment.reindex(columns=["positive","neutral","negative"], fill_value=0)
seat_sentiment = seat_sentiment.reindex(seat_order)

fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(seat_sentiment))
w = 0.26
for i, (col, lbl) in enumerate(zip(["positive","neutral","negative"],
                                    ["Positive","Neutral","Negative"])):
    bars = ax.bar(x + i*w, seat_sentiment[col], width=w,
                  label=lbl, color=COLORS[col], edgecolor="white", linewidth=1.2)
    for bar in bars:
        if bar.get_height() > 5:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    str(int(bar.get_height())), ha="center", fontsize=8)
ax.set_xticks(x + w)
ax.set_xticklabels(seat_sentiment.index, fontsize=11)
ax.set_title("Sentiment Distribution by Seat Type")
ax.set_xlabel("Seat Type"); ax.set_ylabel("Number of Reviews")
ax.legend(title="Sentiment")
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "05_sentiment_by_seat_type.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BIEU DO 6 — MODEL COMPARISON
# ============================================================
fig, ax = plt.subplots(figsize=(10, 6))
model_names = list(results.keys())
accs        = list(results.values())
palette     = ["#3498db","#9b59b6","#1abc9c","#e67e22","#e74c3c"]
bars = ax.bar(model_names, accs, color=palette, width=0.5, edgecolor="white", linewidth=2)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.006,
            f"{acc:.3f}", ha="center", fontsize=12, fontweight="bold")
ax.axhline(y=max(accs), color="red", linestyle="--", alpha=0.5,
           label=f"Best: {max(accs):.3f}")
ax.set_title("Sentiment Model Comparison — Accuracy")
ax.set_ylabel("Accuracy Score"); ax.set_ylim(0, 1.12)
ax.legend()
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "06_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BIEU DO 7 — CONFUSION MATRIX (Best = SVM)
# ============================================================
pred_svm = models["SVM"].predict(X_test)
cm = confusion_matrix(y_test, pred_svm, labels=["negative","neutral","positive"])

fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative","Neutral","Positive"],
            yticklabels=["Negative","Neutral","Positive"],
            linewidths=1, linecolor="white",
            annot_kws={"size": 14, "weight": "bold"}, ax=ax)
ax.set_title("Confusion Matrix — Support Vector Machine")
ax.set_xlabel("Predicted Label"); ax.set_ylabel("True Label")
plt.tight_layout()
plt.savefig(OUT + "07_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# BIEU DO 8 — SCATTER + REGRESSION LINE
# ============================================================
sentiment_map = {"negative": 0, "neutral": 1, "positive": 2}
data["sentiment_num"] = data["sentiment"].map(sentiment_map)

X_reg = data["vader_score"].values.reshape(-1, 1)
y_reg = data["sentiment_num"].values

reg_model = LinearRegression()
reg_model.fit(X_reg, y_reg)
x_line = np.linspace(X_reg.min(), X_reg.max(), 100).reshape(-1, 1)
y_line = reg_model.predict(x_line)

fig, ax = plt.subplots(figsize=(8, 6))
for lbl, num in zip(["negative","neutral","positive"], [0, 1, 2]):
    mask = data["sentiment"] == lbl
    ax.scatter(data.loc[mask, "vader_score"], data.loc[mask, "sentiment_num"],
               color=COLORS[lbl], alpha=0.3, s=15, label=lbl.capitalize())
ax.plot(x_line, y_line, color="black", linewidth=2, label="Regression Line")
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["Negative", "Neutral", "Positive"])
ax.set_xlabel("Lexicon Sentiment Score")
ax.set_ylabel("Sentiment Category")
ax.set_title("Relationship Between Lexicon Score and Sentiment")
ax.legend()
for sp in ["top","right"]: ax.spines[sp].set_visible(False)
plt.tight_layout()
plt.savefig(OUT + "08_regression_scatter.png", dpi=150, bbox_inches="tight")
plt.show()

# ============================================================
# FINAL REPORT
# ============================================================
print("\n" + "="*55)
print("        FINAL REPORT — QATAR AIRWAYS SENTIMENT")
print("="*55)
print(f"  Total reviews analyzed : {len(data)}")
for lbl in ["positive","neutral","negative"]:
    n = counts.get(lbl, 0)
    print(f"  {lbl.capitalize():<12}         : {n} ({n/len(data)*100:.1f}%)")
print("-"*55)
print("  MODEL ACCURACY:")
for m, s in results.items():
    star = " <== BEST" if s == max(results.values()) else ""
    print(f"  {m:<30}: {s:.3f}{star}")
print("="*55)
