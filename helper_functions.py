import os
import numpy as np
import pandas as pd
import xml.etree.ElementTree as et
import string
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import nltk
from nltk import FreqDist, pos_tag, word_tokenize, bigrams
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from wordcloud import STOPWORDS

# get a set of English stop words from nltk and wordcloud
STOP_WORDS = set(stopwords.words('english') + list(STOPWORDS))
doc_pos = {}        # part-of-speech dictionary for documents
topic_pos = {}      # part-of-speech dictionary for topics

def extract_text_from_xml(directory, columns):
    """
    Extract text from all xml documents in a directory.

    Args:
        directory: directory that contains xml files to be extracted

    Returns:
        a data frame of documents

    Example:
        extract_text('datasets')
        Out: docs
    """
    
    docs = []                       # a list of tuple that contain document's id and extracted text
    files = os.listdir(directory)   # get a list of files in the directory
    
    # extract text from xml file
    for f in files:
        doc_id = f[:-4]             # document's id obtained from file's name excluding file's extenstion
        text = []                   # store extracted text for a single document
        tree = et.parse(directory + '/' + f)
        root = tree.getroot()
        for child in root:
            if child.text != doc_id:
                s = child.text
                s.replace(r'[\n\t]', ' ')           # replace newline and tab with a space
                text.append(s.strip())
        
        docs.append((doc_id, ' '.join(text)))      # store text in docs list

    return pd.DataFrame.from_records(docs, columns=columns)

def convert_tag(tag):
    """
    Convert the tag given by nltk.pos_tag to the tag used by wordnet.synsets
    
    Args:
        tag: nltk pos-tag
    
    Returns:
        WordNet pos-tag if tag is found
        'None' if tag is not found
    
    WordNet pos-tag's definition:
        n: noun
        a: adjective
        r: adverb
        v: verb
    """
    
    tag_dict = {'N': 'n', 'J': 'a', 'R': 'r', 'V': 'v'}
    
    try:
        return tag_dict[tag[0]]
    except KeyError:
        return 'None'
		
def preprocess_text(key, text):
    """
    Preprocess text by performing the followings:
        Convert text to lowercase
        Tokenize
        Pos-tag
        Convert NLTK pos-tag to WordNet pos-tag
        Remove stop words and punctuations
        Lemmatize tokens based on part-of-speech tag
        Update global part-of-speech dictionary

    Args:
        key: document's id
        text: text to be processed
    
    Global variables to updated:
        doc_pos: pos dictionary for documents
        topic_pos: pos dictionary for topics

    Returns:
        a list of lemmatized tokens

    Example:
        preprocess_text('topic1', 'Research on Lung cancer treatment for women aged 40 or older; middle-aged.')
        
        Out: ['research', 'lung', 'cancer', 'treatment', 'woman', 'age', 'old', 'middle-aged']
        
        Updated part-of-speech dictionary for documents/topics:
            {'topic1': {('aged', 'v'), ('cancer', 'n'), ('lung', 'n'), ('middle-aged', 'a'),
                          ('older', 'a'), ('research', 'n'), ('treatment', 'n'), ('women', 'n')}}
    """
    
    global doc_pos, topic_pos
    global STOP_WORDS
    
    # tokenize and perform pos-tagging 
    pos = pos_tag(word_tokenize(text.lower()))
    
    # remove stop words, punchtuation, and any pos-tags not found in WordNet's pos
    # remove words with length < 2
    pos = [(tag[0], convert_tag(tag[1])) for tag in pos if tag[0] not in STOP_WORDS \
           and tag[0] not in string.punctuation and tag[1] not in ['.', ':', 'DT'] \
           and convert_tag(tag[1]) != 'None' and len(tag[0]) > 2 and tag[0] != 'none']
    
    # save pos-tag into a dictionary for corresponding document or topic
    if 'topic' in key:
        topic_pos.update({key: set(pos)})
    else:
        doc_pos.update({key: set(pos)})
    
    # lemmatize token based on pos-tag
    lemmatizer = WordNetLemmatizer()
    lemmas = [lemmatizer.lemmatize(tag[0], pos=tag[1]) for tag in pos]
            
    return lemmas

def get_topics_info(num_topics, scores, topic_names, col_labels):
    """
    Create a data frame that store information for query topics such as dominant topic/cluster 
    and corresponding score.

    Args:
        num_topics: number of generated topics/clusters
        scores: a list of dictionary of scores for the generated topics/clusters
        topic_names: a dictionary of topic names
        col_labels: a list of 2 column labels (['Dominant Cluster', 'Probability'])
    
    Returns:
        a data frame with 4 columns: Topic #, Topic Name, Dominant Cluster, Probability
            (The last 2 column labels are defined by the user).
    """
    
    topic_list = []
    name_list = []
    dominant_list = []
    score_list = []
    for i in range(num_topics):
        max_score = max(zip(scores[i].values(), scores[i].keys()))[0]
        for key in scores[i]:
            if scores[i][key] == max_score:
                topic_list.append(i+1)
                name_list.append(topic_names['topic'+str(i+1)])
                dominant_list.append(key)
                score_list.append(max_score)

    # create the data frame
    df = pd.DataFrame()
    df['Topic #'] = topic_list
    df['Topic Name'] = name_list
    df[col_labels[0]] = dominant_list
    df[col_labels[1]] = score_list

    return df

def doc_to_synsets(pos):
    """
    Finds the first synset for each word/tag combination.
    If a synset is not found for that combination it is skipped.

    Args:
        pos: a list of pos-tags tuple

    Returns:
        a list of synsets

    Example:
        doc_to_synsets([('aged', 'v'), ('cancer', 'n'), ('lung', 'n'), ('middle-aged', 'a'), 
                        ('older', 'a'), ('research', 'n'), ('treatment', 'n'), ('women', 'n')])
        Out: [Synset('age.v.01'), Synset('cancer.n.01'), Synset('lung.n.01'), Synset('middle-aged.s.01'), 
                Synset('aged.s.01'), Synset('research.n.01'), Synset('treatment.n.01'), Synset('woman.n.01')]
    """

    synsets = list()
    
    # find the first synset for each word/tag combination
    for tup in pos:
        synset = wordnet.synsets(tup[0], tup[1])
        if len(synset) > 0:
            synsets.append(synset[0])
    
    return synsets

def similarity_score(s1, s2):
    """
    Calculate the normalized similarity score of s1 onto s2

    For each synset in s1, finds the synset in s2 with the largest similarity value.
    Sum of all of the largest similarity values and normalize this value by dividing it by the
    number of largest similarity values found.

    Args:
        s1, s2: list of synsets from doc_to_synsets

    Returns:
        normalized similarity score of s1 onto s2

    Example:
        similarity_score(synsets1, synsets2)
        Out: 0.73333333333333339
    """
    
    # find the largest similarity score
    max_scores = list()
    for synset1 in s1:
        scores=[x for x in [synset1.path_similarity(synset2) for synset2 in s2] if x is not None]
        if scores:
            max_scores.append(max(scores))
    
    return sum(max_scores) / len(max_scores)

def document_path_similarity(doc_id, topic_synsets):
    """Finds the symmetrical similarity between document and topic"""
    
    global doc_pos
    synsets = doc_to_synsets(doc_pos[doc_id])

    return (similarity_score(synsets, topic_synsets) + similarity_score(topic_synsets, synsets)) / 2

def plot_freq_dist(texts, n_gram=1, num_words=25):
    """
    Create a plot of frequency distribution of the most common terms found in the documents.

    Args:
        texts: string of texts
		n_gram: default value to one-gram
        num_words: number of words to be shown, defaulted to 25
    """
	
    temp = texts.split(' ')      # tokenize texts
    if n_gram == 2:
        temp = bigrams(temp)     # create a list of bigrams
	
    fdist = FreqDist(temp)       # FreqDist object

    # set up plot
    plt.figure(figsize=(17, 7))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.xlabel('', fontsize=18)
    plt.ylabel('', fontsize=18)
    
    # plot data
    fdist.plot(num_words)

def plot_docs_distribution(df, col_name, col_score_name, num_topics, topic_labels, trow_label, title,
                           width=0.3, tscale_x=1.5, tscale_y=2, num_cuts=4, pad=90,
                           table_vals = [['Below 0.25', '0.25 to 0.5', '0.5 to 0.75', '0.75 or Above']],
                           bins=[0, 0.25, 0.5, 0.75, 1], 
                           bin_labels=['Very Low','Low','Medium', 'High']):
    """ 
    Create a bar chart of documents distribution per score range.

    Args:
        df: a data frame
        col_name: df's column name of class label
        col_score_name: df's colum name of scores
        num_topics: number of topics/clusters
        trow_label: row labels for data table
        title: plot's title
        width: bar's width
        tscale_x: data table's scaled value for width
        tscale_y: data table's scaled value for height
        num_cuts: number of cuts for bin
        pad: the padding of title above the plot
        bins: a list of score's range
        bin_labels: a list of labels for score ranges
    """
    
    # plot number of documents per score range
    fig = plt.figure(figsize=(12, 7))
    ax = fig.add_subplot(111)

    ind = np.arange(num_cuts)          # the x locations for the groups

    # get the counts of documents
    class_counts = dict(df[col_name].value_counts())

    # plot the bars
    for i in range(num_topics):
        topic_label = topic_labels[i]    # model topic's label

        # get scores
        m_class = df[df[col_name] == topic_label][col_score_name]

        # assign scores to bin
        class_array = np.histogram(m_class, bins=bins)

        # plot the bars
        rects = ax.bar(ind + (width * i), class_array[0], width=width, align='center', 
                    label=col_name + ' ' + topic_label +' (' + str(class_counts[topic_label]) + ')')

        # put value on top of each bar
        for rect in rects:
            h = rect.get_height()
            if h > 0:
                ax.text(rect.get_x()+rect.get_width()/2., 1.01*h, '%d'%int(h), ha='center', va='bottom')

    # show data table
    cell_colors = [['lightblue', 'lightblue', 'lightblue', 'lightblue']]
    table = plt.table(cellText=table_vals, cellColours=cell_colors, colWidths=[0.1] * 6,
                      rowLabels=trow_label, colLabels=bin_labels, rowColours=['lightblue'],
                      loc='top')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(tscale_x, tscale_y)

    # adjust layout to make room for the table:
    plt.subplots_adjust(left=0.2, bottom=0.2)

    # hide top and right border
    _ = [plt.gca().spines[loc].set_visible(False) for loc in ['top', 'right', 'left']]

    ax.set_xticks(ind+width)
    ax.set_xticklabels(tuple(bin_labels))
    #ax.set_xlabel(col_score_name, fontsize=13, fontweight='bold')
    ax.get_yaxis().set_visible(False)

    plt.legend(frameon=False)
    plt.title(title, fontsize=15, verticalalignment='top', pad=pad, fontweight='bold')
    plt.savefig('images/dist_per_' + '_'.join(col_score_name.split(' ')))
    plt.show()

def plot_clusters_3D(x, y, X1, cluster_names, topic_labels, probs, topic_names, cluster_dict, topic_index=0):
    """ 
    Create a 3D plot of clusters and a query topic.

    Args:
        x: documents' features
        y: documents' cluster-labels
        X1: topic's features
        cluster_names: a list of cluster names
        topic_labels: a list of topic labels
        probs: a list of probability score for topic
        topic_names: a dictionary of names for topic
        cluster_dict: a dictionary of top keywords for cluster
        topic_index: index of current topic, default to 0
    """
    
    fig = plt.figure(figsize=(15, 10))
    ax = fig.add_subplot(111, projection='3d')
    color = ['green', 'purple', 'cyan', 'magenta', 'red', 'cyan', 'orange']
    cluster_color = {}       # a dictionary of selected color for each cluster
    
    # count number of documents for each cluster
    unique, counts = np.unique(y, return_counts=True)
    cluster_dist = dict(zip(unique, counts))
    
    # plot clusters
    for i in range(x.shape[0]):
        label = cluster_names[y[i]]      # cluster's name
        c = color[int(y[i]-1)]           # cluster's color
        
        if y[i] not in cluster_color:      # show label on legend
            _ = ax.scatter(x[i, 0], x[i, 1], x[i, 2], s=60, marker='o', c=c, alpha=0.4,
                      label='Cluster ' + str(label) + ' (' + str(cluster_dist[y[i]]) + ')')

        else:
            _ = ax.scatter(x[i, 0], x[i, 1], x[i, 2], s=60, marker='o', c=c, alpha=0.2)
        
        cluster_color.update({y[i]: c})  # update cluster_color dictionary

    # plot topic's cluster
    # different color denotes part of topic belong to the cluster that has the same color
    topic = X1[topic_index]
    topic_color = []
    for i, c in enumerate(topic_labels[topic_index]):
        color = cluster_color[c]
        if color not in topic_color:
            _ = ax.scatter(topic[i, 0], topic[i, 1], topic[i, 2], s=60, marker='x', c=color, 
                    label='Topic ' + str(topic_index + 1)  + ' (' + cluster_names[c] + ')')
                           #str(probs[cluster_names[c]]) + ' from ' + cluster_names[c] + ')')
        else:
            _ = ax.scatter(topic[i, 0], topic[i, 1], topic[i, 2], s=60, marker='x', c=color)
        topic_color.append(color)


    # show data table of closest cluster for corresponding query topic
    max_prob = min(zip(probs.values(), probs.keys()))[0]
    row_labels = []
    row_colors = []
    table_vals = []
    for key in probs:
        if probs[key] == max_prob:
            row_labels.append('Closest Cluster ' + key)
            table_values = [', '.join(cluster_dict['Cluster ' + key][:4])]
            table_vals.append(table_values)
            row_colors.append(cluster_color[cluster_names.index(key)])
            max_prob = probs[key]
        
    col_labels = ['Words in Cluster (Top 4s)']
    table = plt.table(cellText=table_vals, colWidths=[0.1] * 6,
                      rowLabels=row_labels, colLabels=col_labels, rowColours=row_colors, loc='top')
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(3.5, 2)

    # set legend and title
    ax.w_xaxis.set_ticklabels([])
    ax.w_yaxis.set_ticklabels([])
    ax.w_zaxis.set_ticklabels([])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.legend(loc='upper left', frameon=False, labelspacing=0.8)
    plt.title('Topic ' + str(topic_index+1) +
                 '\n' + topic_names['topic'+str(topic_index+1)], fontsize=16, fontweight='bold', pad=120)
    plt.savefig('images/cluster_topic' + str(topic_index+1))
    plt.show()
