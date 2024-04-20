import cudf
import cuml
from cuml.feature_extraction.text import TfidfVectorizer
from cuml.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Veri setini yükleme
train_data = cudf.read_csv('train.csv')
test_data = cudf.read_csv('test.csv')

# Metin ön işleme
stop_words = set(stopwords.words('turkish'))  # Türkçe stop words'leri yükleme

def preprocess_text(text):
    # Küçük harfe dönüştür
    text = text.lower()
    # Noktalama işaretlerini ve sayıları kaldır
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    # Tekrar eden boşlukları kaldır
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize etme
    words = word_tokenize(text)
    # Durdurma kelimelerini kaldırma
    filtered_words = [word for word in words if word not in stop_words]
    # Kullanılacaksa, işlenmiş metni birleştirme
    text = ' '.join(filtered_words)
    return text

train_data['text'] = train_data['text'].applymap(preprocess_text)
test_data['text'] = test_data['text'].applymap(preprocess_text)

# TF-IDF vektörlerini oluşturma
vectorizer = TfidfVectorizer(max_features=5000)  # Öznitelik sayısını isteğe bağlı olarak ayarlayabilirsiniz
X_train = vectorizer.fit_transform(train_data['text'])
X_test = vectorizer.transform(test_data['text'])
y_train = train_data['label']
y_test = test_data['label']

# Modelleme: Naive Bayes sınıflandırıcı kullanma
nb_classifier = MultinomialNB()
nb_classifier.fit(X_train, y_train)

# Modeli değerlendirme
y_pred = nb_classifier.predict(X_test)
accuracy = accuracy_score(y_test.to_array(), y_pred.to_array())
print("Test seti doğruluk:", accuracy)
print("\nSınıflandırma raporu:\n", classification_report(y_test.to_array(), y_pred.to_array()))
