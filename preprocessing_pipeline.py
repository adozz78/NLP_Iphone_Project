import pandas as pd 
import re 
import nltk
import string
from collections import Counter
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk.corpus import stopwords
from utils import slang_words
from spellchecker import SpellChecker
pd.options.mode.chained_assignment = None


# preprocessing functions

#remove debris
def remove_debris(text):

    # the numbers are removed 
    text = re.sub('\d+', '', text)
    # remove the debris generated 
    text = text.replace('hr', '')
    
    if text.endswith("READ MORE"):
        return text[:-9]  
    else:
        return text

# remove puncuation
PUNCT_TO_REMOVE = string.punctuation
def remove_punctuation(text: str) -> str:
    """
    Removes punctuation characters from the input text

    Args:
        text (str): The input text from which punctuation characters will be removed

    Returns:
        str: A new string with all punctuation characters removed
    """
    translation_table = str.maketrans('','',PUNCT_TO_REMOVE)
    return text.translate(translation_table)

def correct_spellings(text: str) -> str:
    """
    Correct spelling errors in the input text using a spell checker.

    This function identifies and corrects spelling errors in the input text by utilizing a spell checker
    (presumably the `spell` object). It splits the input text into words, identifies misspelled words,
    and replaces them with their corrected versions. The corrected text is then returned.

    Args:
        text (str): The input text with possible spelling errors.

    Returns:
        str: A new string with spelling errors corrected.

    Example:
        >>> correct_spellings("Ths is an exmple of misspeled wrds.")
        "This is an example of misspelled words."
    """
    spell = SpellChecker()
    corrected_text = []
    # Split the input text into words
    words = text.split()

    for word in words:
        #identify misspelled words in the text
        misspelled_word = spell.unknown([word])

        if misspelled_word:
            # Attempt to correct the word
            corrected_word = spell.correction(word)
            if corrected_word is not None:
                corrected_text.append(corrected_word)
            else:
                # If correction is not found, keep the original word
                corrected_text.append(word)
        else:
            # If the word is not misspelled, keep it as is
            corrected_text.append(word)

    # back into a string
    corrected_text = ' '.join(corrected_text)

    return corrected_text

# remove stopwords
STOPWORDS = set(stopwords.words('english'))
", ".join(stopwords.words('english'))
def remove_stopwords(text: str) -> str:
    """
    Removes stopwords from the input text

    Args: 
        text (str): The input text from which stopwords will be removed

    Returns:
        str: A new string without stopwords
    """
    # Split the text into words
    words = text.split()
    
    # Remove stopwords 
    filtered_words = [word for word in words if word not in STOPWORDS]
    
    # Join the filtered words back into a string
    return ' '.join(filtered_words)

# remove emoji
def remove_emoji(text: str) -> str:
    """
    Removes emojis from the input text

    Args: 
        text (str): The input text to remove emojis from

    Returns:
        str: A next text without emojis
    """
    # define a regular expression pattern
    emoji_pattern = re.compile("[" 
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"  # Miscellaneous symbols
                           u"\U000024C2-\U0001F251"  # Enclosed characters
                           "]+", flags=re.UNICODE)   # '+' signifies that those characters can occur once or more consecutively

    # Your code here:
    # Use the sub() function to remove emoji_pattern from the text
    # https://www.pythontutorial.net/python-regex/python-regex-sub/
    text_without_emoji = emoji_pattern.sub('', text)
    
    return text_without_emoji

# lemmatize words
lemmatizer = WordNetLemmatizer()
def lemmatize_words(text: str) -> str:
    """
    Apply lemmatization to the input string, considering words' POS tags.

    This function lemmatizes words in the input string based on their POS (Part-of-Speech) tags.
    
    Args:
        text (str): The input text to be lemmatized.

    Returns:
        str: A new string with lemmatized words.
    """
    # Initialize a mapping of POS tags to WordNet tags
    wordnet_map = {
        'N': wordnet.NOUN,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'J': wordnet.ADJ
    }

    words = nltk.word_tokenize(text)
    pos_tagged_text = nltk.pos_tag(words)

    lemmatized_words = []
    for word, pos_tag in pos_tagged_text:

        # Get the WordNet tag for the POS tag
        wordnet_tag = wordnet_map.get(pos_tag[0], wordnet.NOUN)
        
        # Lemmatize the word using the WordNet tag
        lemmatized_word = lemmatizer.lemmatize(word, wordnet_tag)
        lemmatized_words.append(lemmatized_word)
    
    # Join the lemmatized words back into a string
    lemmatized_text = ' '.join(lemmatized_words)
    
    return lemmatized_text


# chat words conversion
slang_words_list = slang_words()
chat_words_list = list(slang_words_list.keys())

def chat_words_conversion(text: str) -> str:
    slang_words_list = slang_words()
    chat_words_list = list(slang_words_list.keys())
    new_text = []

    # Split the input text 
    words = text.split()

    for word in words:
        # Check if the word is contained in chat_words_list
        if word.upper() in chat_words_list:
            # Get the translation 
            translation = slang_words_list[word.upper()]
            new_text.append(translation)
        else:
            # add the word 
            new_text.append(word)

    # back into a string
    converted_text = ' '.join(new_text)

    return converted_text

def preprocess_text(text):
    # Apply the preprocessing functions in a specific order
    text = remove_debris(text)
    text = remove_punctuation(text)
    text = correct_spellings(text)
    text = remove_stopwords(text)
    text = remove_emoji(text)
    text = lemmatize_words(text)
    text = chat_words_conversion(text)
    
    return text




