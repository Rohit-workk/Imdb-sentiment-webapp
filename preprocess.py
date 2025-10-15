import nltk
nltk_packages = ['stopwords', 'wordnet', 'omw-1.4', 'punkt', 'averaged_perceptron_tagger']
for package in nltk_packages:
    try:
        nltk.data.find(package)
    except LookupError:
        nltk.download(package)
import re
import string
import contractions
import emoji
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag

negation_words = {"not", "no", "never", "n't"}
stop_words = set(stopwords.words('english')) - negation_words
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {'J': wordnet.ADJ, 'N': wordnet.NOUN, 'V': wordnet.VERB, 'R': wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def preprocess_text(text):
    text = str(text)
    text = re.sub(r'<.*?>', '', text)
    text = contractions.fix(text)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = emoji.demojize(text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.encode('ascii', 'ignore').decode()
    text = re.sub(r'\s+', ' ', text).strip()

    filtered_words = []
    words = text.split()
    for word in words:
        if word not in stop_words:
            pos = get_wordnet_pos(word)
            lemma = lemmatizer.lemmatize(word, pos)
            filtered_words.append(lemma)
    return ' '.join(filtered_words)
