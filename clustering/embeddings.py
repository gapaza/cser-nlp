import os
import config
import importlib.util
import sys
import pickle
import tiktoken
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from nltk import tokenize as nlktokenize

import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster


import openai
openai.api_key = config.openai_api_key



def num_tokens(input):
    encoding = tiktoken.get_encoding("gpt2")
    num_tokens = len(encoding.encode(input))
    return num_tokens

def get_tokens(input):
    encoding = tiktoken.get_encoding("gpt2")
    tokens = encoding.encode(input)
    return tokens

def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    - Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))


class SimilaritySearch:

    def __init__(self):
        self.embedding_model = 'text-embedding-3-small'
        self.embedding_path = os.path.join(config.clustering_dir, 'embeddings.pkl')
        self.clusters_path = os.path.join(config.clustering_dir, 'clusters')
        self.embeddings = {}
        self.read_embedding_store()

    def search(self, query, search_list, top_n=10):
        # embedding_dict = self.context_manager.get_embedding_dict()
        embedding_dict = self.embed_list_batch(search_list)
        query_embedding = self.embed_text(query)
        context_similarities = sorted([
            (vector_similarity(query_embedding, embedded_context), idx) for idx, embedded_context in
            enumerate(embedding_dict)
        ], reverse=True)
        if len(context_similarities) < top_n:
            top_n = len(context_similarities)

        sorted_list = [search_list[i[1]] for i in context_similarities]
        return context_similarities[:top_n], sorted_list[:top_n]

    def read_embedding_store(self):
        if os.path.exists(self.embedding_path):
            self.embeddings = pickle.load(open(self.embedding_path, 'rb'))
        else:
            with open(self.embedding_path, 'wb') as f:
                pickle.dump(self.embeddings, f)

    def write_embedding_store(self):
        with open(self.embedding_path, 'wb') as f:
            pickle.dump(self.embeddings, f)

    # --------------------- #
    # --- Embed Context --- #
    # --------------------- #

    def embed_text(self, text):
        if text in self.embeddings.keys():
            return self.embeddings[text]
        result = openai.Embedding.create(model=self.embedding_model, input=text)
        self.embeddings[text] = result["data"][0]["embedding"]
        self.write_embedding_store()
        return self.embeddings[text]

    def embed_list_batch(self, text_list):
        if all(text in self.embeddings for text in text_list):
            # print('All texts found in pre-cache')
            return [self.embeddings[text] for text in text_list]

        embedding_list = []
        temp_batch = []
        batch_cnt = 1
        for idx, value in enumerate(text_list):
            if value == '':
                embedding_list.append(None)
                continue
            temp_batch.append(value)
            if len(temp_batch) == 1000:
                print('--> BATCH:', batch_cnt, '(1000)')
                batch_result = self.embed_list(temp_batch)
                embedding_list += batch_result
                temp_batch = []
                batch_cnt += 1
        if len(temp_batch) > 0:
            batch_result = self.embed_list(temp_batch)
            embedding_list += batch_result
        self.write_embedding_store()
        return embedding_list

    def embed_list(self, text_list):
        embeddings = {}
        not_found = []

        # Iterate over the text_list
        for text in text_list:
            # If the text is in the embeddings dictionary, retrieve it
            if text in self.embeddings:
                embeddings[text] = self.embeddings[text]
            else:
                # If the text is not in the embeddings dictionary, add it to the not_found list
                print('New text found: ', text)
                not_found.append(text)

        # If there are texts not found in the cache, generate embeddings for them
        if not_found:
            result = openai.embeddings.create(model=self.embedding_model, input=not_found)
            for text, res in zip(not_found, result.data):
                embedding = res.embedding
                # Store the new embedding in the cache and the return dictionary
                self.embeddings[text] = embedding
                embeddings[text] = embedding
        else:
            print('All texts found in cache')

        # Return the embeddings in the order they were requested
        return [embeddings[text] for text in text_list]

    def group_embeddings_kmeans(self, embedding_list, num_clusters=2, plot=False):
        embedded_strings = np.array(embedding_list)
        if plot:
            inertias = []
            K = range(1, 15)
            for k in K:
                kmeans = KMeans(n_clusters=k, random_state=0).fit(embedded_strings)
                inertias.append(kmeans.inertia_)
            plt.plot(K, inertias, 'bx-')
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title('Elbow Method For Optimal k')
            plt.show()
        kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(embedded_strings)
        return kmeans.labels_

    def group_embeddings_hierarchical(self, embedding_list, threshold=3.9):
        # 3.4 / 3.9 is pretty good at identifying nonsense for ~150 pages
        embedded_strings = np.array(embedding_list)
        Z = linkage(embedded_strings, method='ward')
        labels = fcluster(Z, threshold, criterion='distance')
        return labels

    def write_clusters(self, text_list, clusters):
        # First, remove all files in the clusters directory
        for file in os.listdir(self.clusters_path):
            os.remove(os.path.join(self.clusters_path, file))

        # Then, write the new clusters to the clusters directory, where each cluster has a unique file
        for cluster in set(clusters):
            with open(os.path.join(self.clusters_path, f'cluster_{cluster}.txt'), 'w') as f:
                for idx, text in enumerate(text_list):
                    if clusters[idx] == cluster:
                        f.write(text + '\n')

    def plot_embeddings(self, embedding_list, clusters):
        embeddings = np.array(embedding_list)
        tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)

        # Plotting
        plt.figure(figsize=(10, 8))
        # scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.7)
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=clusters, cmap='jet', alpha=0.7)
        plt.colorbar(scatter)
        plt.title('2D Visualization of Sentence Embeddings using t-SNE')
        plt.xlabel('t-SNE feature 1')
        plt.ylabel('t-SNE feature 2')
        plt.show()




if __name__ == '__main__':

    similarity = SimilaritySearch()


    print(len(config.req_data))

    to_embed = config.req_data[:10000]
    embedded = similarity.embed_list_batch(to_embed)

    # clusters = similarity.group_embeddings_kmeans(embedded, num_clusters=15, plot=False)
    clusters = similarity.group_embeddings_hierarchical(embedded, threshold=1.5)


    similarity.write_clusters(to_embed, clusters)
    similarity.plot_embeddings(embedded, clusters)



    print(len(to_embed))
    print(len(embedded))
    print(clusters)
