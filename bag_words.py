"""
Sonia Vaidya
NLP: Bag of Words
Independently learning about some NLP techniques from online resources
(The baseline code is from this Medium article: https://ayselaydin.medium.com/4-bag-of-words-model-in-nlp-434cb38cdd1b,
 I went through all of the concepts and processes used in this program to learn more about NLP in general)
"""
import nltk    # nltk is a library with tools for natural language processing (installation required)
from nltk.stem import WordNetLemmatizer   # WordNetLemmatizer allows us to lemmatize words (it is a built-in morphy function --> see lemmatization notes in Notion)
import re   # re module allows for regular expression support 
from nltk.corpus import stopwords  # stopwords --> a set of commonly used words in a language (e.g. 'the', 'is', 'and', 'of')
from sklearn.feature_extraction.text import CountVectorizer  # sklearn is SciKit Learn, python library for machine learning 
                                                             # feature extraction allows us to extract feature vectors from text documents
                                                             # CountVectorizer is a class that allows us to convert a collection of text documents to a matrix of token counts

def preprocess_text(text: str) -> str:
    # goal: remove stop words, emojis, numbers, punctuation marks, excess spaces 
    # convert all characters to lowercase 
    
    text = text.split()  # Making a list of words in the sentence 
    
    inst_lemma = WordNetLemmatizer()
    
    non_stop_words = [w for w in text if w not in set(stopwords.words('english'))]

    text = [inst_lemma.lemmatize(n) for n in non_stop_words]  # lemmatizes each word that's not a stop word 
    
    text = ' '.join(text)  ## Adding a space between every word in text list, making it a string

    text = re.sub(r'[0-9]+', '', text)   ## r'[0-9]' is a character class for digits 0-9
    ## The '+' is a quantifier that specifies that the preceding character or group ([0-9] in this case) should match one or more times.
    ## It means the pattern should match if there's at least one digit, but it can also match if there are multiple consecutive digits.
    ## We want to remove any digits from the text using this re.sub() method

    text = re.sub(r'[^\w\s]', '', text) 
    ## The ^ represents negation
    ## \w represents all alphanumeric characters
    ## \s represents whitespace characters 
    ## We want to remove any character that is not alphameric or not whitespace 

    ### This is pattern of emojis and misc. ASCII characters, which we will remove later using the re.sub() method below 
    emoji_pattern = r'^(?:[\u2700-\u27bf]|(?:\ud83c[\udde6-\uddff]){1,2}|(?:\ud83d[\udc00-\ude4f]){1,2}|[\ud800-\udbff][\udc00-\udfff]|[\u0021-\u002f\u003a-\u0040\u005b-\u0060\u007b-\u007e]|\u3299|\u3297|\u303d|\u3030|\u24c2|\ud83c[\udd70-\udd71]|\ud83c[\udd7e-\udd7f]|\ud83c\udd8e|\ud83c[\udd91-\udd9a]|\ud83c[\udde6-\uddff]|\ud83c[\ude01-\ude02]|\ud83c\ude1a|\ud83c\ude2f|\ud83c[\ude32-\ude3a]|\ud83c[\ude50-\ude51]|\u203c|\u2049|\u25aa|\u25ab|\u25b6|\u25c0|\u25fb|\u25fc|\u25fd|\u25fe|\u2600|\u2601|\u260e|\u2611|[^\u0000-\u007F])+$'
    text = re.sub(emoji_pattern, '', text)

    text = re.sub(r'[\s+]', ' ', text)  ## replace one or more (+) whitespace characters (\s) in the string text with a single space (' ')
    # whitespace characters may include spaces, tabs, newlines, etc.

    text = text.lower().strip()      # We want to remove all the leading/trailing whitespaces 

    return text 


def process_paragraph(paragraph: str) -> list:
    list_of_sentences = nltk.sent_tokenize(paragraph)  # splits paragraph into sentences (tokenization)
    list_of_processed_sentences = [preprocess_text(sentence) for sentence in list_of_sentences] # preprocesses each sentence by removing whitespaces, punctuation, numbers, emojis, etc.
    return list_of_processed_sentences 


def to_matrix(corpus: list):
    inst_vectorizer = CountVectorizer()
    
    # sklearn allows us to combine the fit and transform processes into one method called fit_transform 
        # fitting: learning parameters from the corpus (key vocabulary, unique words)
        # transformation: modifies input data, converts into matrix of token counts (number of occurrences of each word, arranged in an organized array format)
    matrix = inst_vectorizer.fit_transform(corpus)  

    # ^^^ This function can return a SPARSE MATRIX (default) or DENSE MATRIX, must convert to an array
    """
        Sparse matrix: Memory-efficient way to store matrices, where many of the values are 0
        Dense matrix: Must set the parameter sparse=False, all elements stored explicitly regardless of values, mostly non-zero values in data 
    """
        # The toarray() method converts the sparse matrix into a dense NumPy array
        # Will represent all elements of the matrix, including the zeros, as explicit values in memory
    matrix_array = matrix.toarray()

    unique_words = inst_vectorizer.get_feature_names_out()  ## building a vocabulary of all unique words (or terms) present in the text, each will have an associated index in the matrix
    
    return unique_words, matrix_array

## TRY IT OUT!
corp = process_paragraph("INSERT TEXT HERE!")
words_and_matrix = to_matrix(corp)

print(words_and_matrix[0])
print(words_and_matrix[1])
