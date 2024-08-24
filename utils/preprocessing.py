import pandas as pd
import stanza, re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

class Preprocess:
    @staticmethod
    def remove_non_ascii(df, column_name):
        special_char_pattern = re.compile(r'[^\x00-\x7F]+')
        hash_pattern = re.compile(r'\s#\s')
        def clean_text(value):
            if isinstance(value, str):
                value = special_char_pattern.sub('', value)
                value = hash_pattern.sub(' ', value)
                return value.strip()
            return value

        df[column_name] = df[column_name].apply(clean_text)
        return df
    
    @staticmethod
    def remove_usertag(df, column_name):
        hashtag_pattern = re.compile(r'@\w+')
        def clean_text(value):
            if isinstance(value, str):
                value = hashtag_pattern.sub('', value)
                return value.strip() 
            return value

        df[column_name] = df[column_name].apply(clean_text)
        return df
    
    @staticmethod
    def remove_hashtag(df, column_name):
        hashtag_pattern = re.compile(r'#\w+')
        def clean_text(value):
            if isinstance(value, str):
                value = hashtag_pattern.sub('', value)
                return value.strip() 
            return value

        df[column_name] = df[column_name].apply(clean_text)
        return df

    @staticmethod
    def reformat_hashtag(df, column_name):
        camel_case_pattern = re.compile(r'#([A-Z][a-z]+[A-Z][a-zA-Z]*)')
        hashtag_pattern = re.compile(r'#(\w+)')
        single_hash_pattern = re.compile(r'#')

        def clean_text(value):
            if isinstance(value, str):
                value = camel_case_pattern.sub('', value)
                value = hashtag_pattern.sub(r'\1', value)
                value = single_hash_pattern.sub('', value)
                return value.strip()
            return value

        df[column_name] = df[column_name].apply(clean_text)
        return df

    @staticmethod
    def remove_stop_words(df, column_name, new_col="text_tranformed", transformation=["tokenize", "lemma", "lower"], save_in="../data/transformed/train.lemma.csv"):
        nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos,lemma', verbose=False)
        #stop_words = set(stanza.download('en', processors='tokenize,lemma')._lang.stop_words)
        stop_words = ['.', ',', ';', ':', '-']

        def process_text(value):
            if isinstance(value, str):
                doc = nlp(value)
                
                if "lemma" in transformation and "lower" in transformation:
                    cleaned_text = ' '.join([word.lemma.lower() for word in doc.iter_words() if word.lemma.lower() not in stop_words])
                elif "lemma" in transformation and "lower" not in transformation:
                    cleaned_text = ' '.join([word.lemma for word in doc.iter_words() if word.lemma.lower() not in stop_words])
                elif "lemma" not in transformation and "lower" not in transformation:
                    cleaned_text = ' '.join([word.text for word in doc.iter_words() if word.lemma.lower() not in stop_words])
                else:
                    cleaned_text = ' '.join([word.text.lower() for word in doc.iter_words() if word.lemma.lower() not in stop_words])
                return cleaned_text
            return value

        df.loc[:, new_col] = df[column_name].progress_apply(process_text)
        df.to_csv(save_in, index=False)
        return df

    def build_bilou_encoding(df, text_col="text", save_in="../data/transformed/train.tag.csv"):
        bilou_data = []
        for _, row in df.iterrows():
            tweet_id = row['tweet_id']
            text = row[text_col]
            location_mentions = row['location_mentions']

            # Split location mentions
            if pd.notna(location_mentions):
                locations = [loc.split('=>') for loc in location_mentions.split(' * ')]
            else:
                locations = []

            loc_dict = {loc[0].strip(): loc[1].strip().split(' ')[0] for loc in locations}
            tokens = re.findall(r'\b\w+\b', text)
            tags = ['O'] * len(tokens)

            # Assign BILOU tags based on location mentions
            i = 0
            while i < len(tokens):
                token = tokens[i]
                if token in loc_dict:
                    tag_type = loc_dict[token]
                    if i == 0 or tokens[i - 1] not in loc_dict:
                        tags[i] = 'U-' + tag_type  # Unit entity
                    else:
                        # Check if it's the last token of an entity
                        if i == len(tokens) - 1 or tokens[i + 1] not in loc_dict:
                            tags[i] = 'L-' + tag_type  # Last token of an entity
                        else:
                            tags[i] = 'B-' + tag_type  # Beginning of an entity
                            # Tag all following tokens that are the same entity as Inside
                            while i < len(tokens) and tokens[i] == token:
                                tags[i] = 'I-' + tag_type
                                i += 1
                            continue
                i += 1

            for token, tag in zip(tokens, tags):
                bilou_data.append({'sentence_id': tweet_id, 'words': token, 'labels': tag})

        bilou_df = pd.DataFrame(bilou_data)
        df.to_csv(save_in, index=False)
        return bilou_df









    @staticmethod
    def remove_stop_words_(text):
        stanza.download('en', verbose=False)
        nlp = stanza.Pipeline('en', tokenize_pretokenized=True, verbose=False)
        stop_words = set(stopwords.words('english'))
        doc = nlp(text)
        filtered_words = [word.text for sent in doc.sentences for word in sent.words if word.text.lower() not in stop_words]
        return ' '.join(filtered_words)

    
    @staticmethod
    def generate_bio_tags(text, location):
        nlp = stanza.Pipeline(lang='en', processors='tokenize', verbose=False)
        doc = nlp(text)
        tokens = []
        tags = []
        loc_words = location.split()
        
        for sentence in doc.sentences:
            for i, token in enumerate(sentence.tokens):
                token_text = token.text
                tokens.append(token_text)
                
                if token_text in loc_words:
                    if loc_words[0] == token_text:
                        tags.append('B-geo')
                    else:
                        tags.append('I-geo')
                else:
                    tags.append('O')
        
        return tokens, tags
    
    
    @staticmethod
    def treat_hashtags(text):
        """If hashtag is all uppercase, leave it as is, else split into words(CamelCase)"""
        def replace_hashtag(match):
            hashtag = match.group(0)[1:]
            if hashtag.isupper():
                return hashtag
            else:
                return re.sub(r'(?<!^)(?=[A-Z])', ' ', hashtag)

        return re.sub(r'#\w+', replace_hashtag, text)
    
    # @staticmethod
    # def remove_stop_words(text):
    #     stanza.download('en', verbose=False)
    #     nlp = stanza.Pipeline('en', tokenize_pretokenized=True, verbose=False)
    #     stop_words = set(stopwords.words('english'))
    #     doc = nlp(text)
    #     filtered_words = [word.text for sent in doc.sentences for word in sent.words if word.text.lower() not in stop_words]
    #     return ' '.join(filtered_words)
    
    @staticmethod
    def correct_spelling(text):
        spell = SpellChecker()
        words = text.split()
        corrected_words = []
        for word in words:
            candidates = spell.candidates(word)
            if candidates:
                corrected_word = spell.candidates(word).pop()
            else:
                corrected_word = word
            corrected_words.append(corrected_word)
        corrected_text = ' '.join(corrected_words)
        return corrected_text