import nltk
import re
import string
import contractions
import emoji
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

# -----------------------------
# Auto-download required NLTK resources
# -----------------------------
nltk_packages = {
    'stopwords': 'corpora/stopwords',
    'wordnet': 'corpora/wordnet',
    'omw-1.4': 'corpora/omw-1.4',
    'punkt': 'tokenizers/punkt',
    'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger'
}

for pkg, path in nltk_packages.items():
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg)

# -----------------------------
# Setup for preprocessing
# -----------------------------
negation_words = {"not", "no", "never", "n't"}
stop_words = set(stopwords.words('english')) - negation_words
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Function to map POS tags for lemmatizer
# -----------------------------
def get_wordnet_pos(word):
    # Use 'lang=eng' to avoid _eng issue
    tag = pos_tag([word], lang='eng')[0][1][0].upper()
    tag_dict = {
        'J': wordnet.ADJ,
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)

# -----------------------------
# Main preprocessing function
# -----------------------------
def preprocess_text(text):
    text = str(text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Expand contractions (e.g., "don't" -> "do not")
    text = contractions.fix(text)
    # Remove URLs and mentions
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    # Lowercase
    text = text.lower()
    # Convert emojis to text
    text = emoji.demojize(text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Encode to ASCII
    text = text.encode('ascii', 'ignore').decode()
    # Remove extra whitespaces
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize, remove stopwords, and lemmatize
    filtered_words = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            pos = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, pos)
            filtered_words.append(lemma)

    return ' '.join(filtered_words)
