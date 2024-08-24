import os
import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
import time
import base64
import json
from dotenv import load_dotenv

load_dotenv()
token = os.getenv('GIT_TOKEN')

class LMR_JSON_Scrapper:
    def __init__(self, feed_json_path="../utils/feed-json.json", output_dir="../data/self_scrapped/raw/"):
        with open(feed_json_path, 'r') as f:
            self.feed_data = json.load(f)
        self.output_dir = output_dir
        self.token = token
        self.csv_files = {
            'train': os.path.join(output_dir, 'train.<tag>.csv'),
            'test_unlabeled': os.path.join(output_dir, 'test_unlabeled.<tag>.csv'),
            'dev': os.path.join(output_dir, 'dev.<tag>.csv')
        }
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def fetch_file_urls(self, tree_url):
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
        response = requests.get(tree_url, headers=headers)
        response.raise_for_status()
        tree_data = response.json()
        file_urls = {item['path'].split('.')[0]: item['url'] for item in tree_data['tree'] if item['path'] in ['dev.jsonl', 'test_unlabeled.jsonl', 'train.jsonl']}
        return file_urls

    def fetch_file_content(self, blob_url):
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
        response = requests.get(blob_url, headers=headers)
        response.raise_for_status()
        blob_data = response.json()
        content_base64 = blob_data['content']
        content = base64.b64decode(content_base64).decode('utf-8')
        return content

    def process_file(self, content, df_id):
        entries = content.strip().splitlines()
        data = []
        for entry in entries:
            entry_dict = json.loads(entry)
            tweet_id = entry_dict.get('tweet_id', '')
            text = entry_dict.get('text', '')
            location_mentions = entry_dict.get('location_mentions', [])
            locations = ' * '.join([f"{loc['text']}=>{loc['type'].split('/')[0].upper()}" for loc in location_mentions])
            data.append({'tweet_id': 'ID_' + str(tweet_id), 'text': text, 'location_mentions': locations})
        
        df = pd.DataFrame(data)
        return df

    def save_to_csv(self, df, file_path):
        df.to_csv(file_path, index=False)

    def run(self):
        for tree_item in self.feed_data['tree']:
            tag = tree_item['path']
            tree_url = tree_item['url']
            print(f"Processing dataset: {tag}")

            file_urls = self.fetch_file_urls(tree_url)

            for file_type, file_url in tqdm(file_urls.items(), desc=f"Extracting Files ", unit="file"):
                content = self.fetch_file_content(file_url)
                data = self.process_file(content, tag)
                self.save_to_csv(data, self.csv_files[file_type].replace('<tag>', tag))

        print("Processing complete.")

class LMR_BILOU_Scrapper:
    def __init__(self, feed_json_path="../utils/feed-bilou.json", output_dir="../data/self_scrapped/bilou/"):
        with open(feed_json_path, 'r') as f:
            self.feed_data = json.load(f)
        self.output_dir = output_dir
        self.token = token
        self.csv_files = {
            'train': os.path.join(output_dir, 'train.<tag>.csv'),
            'dev': os.path.join(output_dir, 'dev.<tag>.csv')
        }
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def fetch_file_urls(self, tree_url):
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
        response = requests.get(tree_url, headers=headers)
        response.raise_for_status()
        tree_data = response.json()
        file_urls = {item['path'].split('.')[0]: item['url'] for item in tree_data['tree'] if item['path'] in ['train.txt', 'dev.txt']}
        return file_urls

    def fetch_file_content(self, blob_url):
        headers = {}
        if self.token:
            headers['Authorization'] = f'token {self.token}'
        response = requests.get(blob_url, headers=headers)
        response.raise_for_status()
        blob_data = response.json()
        content_base64 = blob_data['content']
        content = base64.b64decode(content_base64).decode('utf-8')
        return content

    def process_file(self, content, df_id):
        sentences = content.strip().split('\n\n')
        data = []
        for id_sentence, sentence in enumerate(sentences):
            lines = sentence.split('\n')
            for line in lines:
                if line.strip():
                    word, tag = line.rsplit(' ', 1)
                    data.append({'id_sentence': str(df_id).upper() + "_" + str(id_sentence), 'word': word, 'tag': tag})
        return data

    def save_to_csv(self, data, file_path):
        df = pd.DataFrame(data)
        df.to_csv(file_path, index=False)

    def run(self):
        for tree_item in self.feed_data['tree']:
            tag = tree_item['path']
            tree_url = tree_item['url']
            print(f"Processing dataset: {tag}")

            file_urls = self.fetch_file_urls(tree_url)

            for file_type, file_url in tqdm(file_urls.items(), desc=f"Extracting Files ", unit="file"):
                content = self.fetch_file_content(file_url)
                data = self.process_file(content, tag)
                self.save_to_csv(data, self.csv_files[file_type].replace('<tag>', tag))

        print("Processing complete.")
        
class Predictions:
    @staticmethod
    def to_csv(ids, predictions, folder='../submissions'):
        if not os.path.exists(folder):
            os.makedirs(folder)

        existing_files = [f for f in os.listdir(folder) if f.startswith('submission_') and f.endswith('.csv')]
        if existing_files:
            indices = [int(f.split('_')[1].split('.')[0]) for f in existing_files]
            next_index = max(indices) + 1
        else:
            next_index = 1
        
        filename = f'submission_{next_index}.csv'
        filepath = os.path.join(folder, filename)
        df = pd.DataFrame({'tweet_id': ids, 'location': predictions})
        df.to_csv(filepath, index=False)
        print(f'Saved predictions to {filepath}')