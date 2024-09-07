import pandas as pd
import stanza, re
from spellchecker import SpellChecker
from nltk.corpus import stopwords
from tqdm import tqdm
tqdm.pandas()

class Preprocess:
    @staticmethod
    def remove_prefix(df, df_type="train", text_column='text', location_column='location'):
        if df_type == "train":
            pattern = r"^[^:]*:"
            def clean_text(row):
                text = str(row[text_column])
                location_words = set(str(row[location_column]).split())

                match = re.match(pattern, text)
                if match:
                    prefix = match.group(0)
                    prefix_words = set(prefix.split())
                    if not location_words & prefix_words:
                        text = re.sub(pattern, "", text).strip()
                return text

            df[text_column] = df.apply(clean_text, axis=1)
            return df
        else:
            pattern = r"^[^:]*:"
            def clean_text(text):
                match = re.match(pattern, text)
                if match:
                    prefix = match.group(0)
                    prefix_words = prefix.split()
                    if len(prefix_words) < 3:
                        text = re.sub(pattern, "", text).strip()
                return text
            df[text_column] = df[text_column].apply(lambda x: clean_text(str(x)))
            return df
    
    @staticmethod
    def reformat_useless_char(df, column_name='text'):
        characters_to_remove = r"[()\[\]{},;#\"]"
        df[column_name] = df[column_name].apply(lambda x: re.sub(characters_to_remove, "", str(x)))
        return df
    
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
        stop_words = ['', ',', ';', ':', '-']

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

        df2 = df.copy()
        df2.loc[:, new_col] = df[column_name].progress_apply(process_text)
        df2.to_csv(save_in, index=False)
        return df2

    def build_bilou_encoding(df, text_col="text", save_in="../data/transformed/train.tag.csv"):
        """
        B-<type>: Beginning of an entity.
        I-<type>: Inside an entity.
        L-<type>: Last token of an entity.
        U-<type>: Unit entity (a single token entity).
        O: Outside any entity.
        """
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

            # loc_dict = {loc[0].strip().lower(): loc[1].strip().split(' ')[0] for loc in locations}
            loc_dict = {}
            for loc in locations:
                loc_name = loc[0].strip().lower()
                loc_type = loc[1].strip().split(' ')[0]
                loc_tokens = re.findall(r'\b\w+\b', loc_name)
                for token in loc_tokens:
                    loc_dict[token] = loc_type

            tokens = re.findall(r'\b\w+\b', text)
            tags = ['O'] * len(tokens)
            i = 0
            while i < len(tokens):
                token = tokens[i].lower()
                if token in loc_dict:
                    tag_type = loc_dict[token]

                    # If the token is the first or not preceded by the same entity, or the next token is of a different type
                    if (i == 0 or tokens[i - 1] not in loc_dict or loc_dict[tokens[i - 1]] != tag_type) and \
                    (i == len(tokens) - 1 or tokens[i + 1] not in loc_dict or loc_dict[tokens[i + 1]] != tag_type):
                        tags[i] = 'U-' + tag_type  # Unit entity
                    else:
                        # Beginning of a multi-token entity
                        if i == 0 or tokens[i - 1] not in loc_dict or loc_dict[tokens[i - 1]] != tag_type:
                            tags[i] = 'B-' + tag_type
                        # Inside a multi-token entity
                        elif i < len(tokens) - 1 and tokens[i + 1] in loc_dict and loc_dict[tokens[i + 1]] == tag_type:
                            tags[i] = 'I-' + tag_type
                        # Last token of a multi-token entity
                        elif i == len(tokens) - 1 or tokens[i + 1] not in loc_dict or loc_dict[tokens[i + 1]] != tag_type:
                            tags[i] = 'L-' + tag_type
                i += 1

            for token, tag in zip(tokens, tags):
                bilou_data.append({'sentence_id': tweet_id, 'words': token, 'labels': tag})

        bilou_df = pd.DataFrame(bilou_data)
        bilou_df.to_csv(save_in, index=False)
        return bilou_df
    
    @staticmethod
    def build_bio_encoding(df, text_col="text", save_in="../data/transformed/train.tag.csv"):
        """
        B-<type>: Beginning of an entity.
        I-<type>: Inside an entity.
        O: Outside any entity.
        """
        bio_data = []
        for _, row in df.iterrows():
            tweet_id = row['tweet_id']
            text = row[text_col]
            location_mentions = row['location_mentions']

            # Split location mentions
            if pd.notna(location_mentions):
                locations = [loc.split('=>') for loc in location_mentions.split(' * ')]
            else:
                locations = []

            # Create a dictionary of location mentions
            loc_dict = {}
            for loc in locations:
                loc_name = loc[0].strip().lower()
                loc_type = loc[1].strip().split(' ')[0]
                loc_tokens = re.findall(r'\b\w+\b', loc_name)
                for token in loc_tokens:
                    loc_dict[token] = loc_type

            tokens = re.findall(r'\b\w+\b', text)
            tags = ['O'] * len(tokens)
            i = 0

            print(loc_dict)
            print(tokens)

            while i < len(tokens):
                token = tokens[i].lower()
                if token in loc_dict:
                    tag_type = loc_dict[token]

                    # Beginning of a multi-token entity
                    if i == 0 or tokens[i - 1] not in loc_dict or loc_dict[tokens[i - 1]] != tag_type:
                        tags[i] = 'B-' + tag_type
                    else:
                        # Inside a multi-token entity
                        tags[i] = 'I-' + tag_type
                i += 1

            for token, tag in zip(tokens, tags):
                bio_data.append({'sentence_id': tweet_id, 'words': token, 'labels': tag})

        bio_df = pd.DataFrame(bio_data)
        bio_df.to_csv(save_in, index=False)
        return bio_df
    
    @staticmethod
    def build_bio_encoding(df, text_col="text", save_in="../data/transformed/train.tag.csv"):
        """
        B-<type>: Beginning of an entity.
        I-<type>: Inside an entity.
        O: Outside any entity.
        """
        bio_data = []
        for _, row in df.iterrows():
            tweet_id = row['tweet_id']
            text = row[text_col]
            location_mentions = row['location_mentions']

            # Split location mentions
            if pd.notna(location_mentions):
                locations = [loc.split('=>') for loc in location_mentions.split(' * ')]
            else:
                locations = []

            # Create a dictionary of location mentions
            loc_dict = {}
            for loc in locations:
                loc_name = loc[0].strip().lower()
                loc_type = loc[1].strip().split(' ')[0]
                loc_tokens = re.findall(r'\b\w+\b', loc_name)
                loc_dict[tuple(loc_tokens)] = loc_type

            tokens = re.findall(r'\b\w+\b', text)
            tags = ['O'] * len(tokens)
            i = 0

            while i < len(tokens):
                found_match = False
                for length in range(len(tokens) - i, 0, -1):  # Check longer sequences first
                    token_seq = tuple(tokens[i:i + length])
                    token_seq_lower = tuple(map(str.lower, token_seq))
                    if token_seq_lower in loc_dict:
                        tag_type = loc_dict[token_seq_lower]
                        tags[i] = 'B-' + tag_type
                        if length > 1:
                            for j in range(1, length):
                                tags[i + j] = 'I-' + tag_type
                        i += length
                        found_match = True
                        break
                if not found_match:
                    i += 1

            for token, tag in zip(tokens, tags):
                bio_data.append({'sentence_id': tweet_id, 'words': token, 'labels': tag})

        bio_df = pd.DataFrame(bio_data)
        bio_df.to_csv(save_in, index=False)
        return bio_df

    @staticmethod
    def build_io_encoding(df, text_col="text", save_in="../data/transformed/train.tag.csv"):
        """
        I-<type>: Inside an entity.
        O: Outside any entity.
        """
        io_data = []
        for _, row in df.iterrows():
            tweet_id = row['tweet_id']
            text = row[text_col]
            location_mentions = row['location_mentions']

            # Split location mentions
            if pd.notna(location_mentions):
                locations = [loc.split('=>') for loc in location_mentions.split(' * ')]
            else:
                locations = []

            # Create a dictionary of location mentions
            loc_dict = {}
            for loc in locations:
                loc_name = loc[0].strip().lower()
                loc_type = loc[1].strip().split(' ')[0]
                loc_tokens = re.findall(r'\b\w+\b', loc_name)
                for token in loc_tokens:
                    loc_dict[token] = loc_type

            tokens = re.findall(r'\b\w+\b', text)
            tags = ['O'] * len(tokens)
            for i, token in enumerate(tokens):
                token_lower = token.lower()
                if token_lower in loc_dict:
                    tags[i] = 'I-' + loc_dict[token_lower]

            for token, tag in zip(tokens, tags):
                io_data.append({'sentence_id': tweet_id, 'words': token, 'labels': tag})

        io_df = pd.DataFrame(io_data)
        io_df.to_csv(save_in, index=False)
        return io_df
    
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