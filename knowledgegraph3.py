#!/usr/bin/env python
# coding: utf-8

# In[2]:


# custom knowedgegraph


# In[1]:


import pandas as pd
import re
import spacy
import neuralcoref

from spacy.matcher import Matcher 
from spacy.tokens import Span


# In[2]:


nlp = spacy.load('en_core_web_lg')
neuralcoref.add_to_pipe(nlp)


# In[3]:


# we will make different functions for different tasks
# def get_entities(): to get the subject and object
# def get_relation(): to get the realation of between the subject and obect of the sentence


# In[4]:


def refine_ent(ent, sent):
    unwanted_tokens = (
        'PRON',  # pronouns
        'PART',  # particle
        'DET',  # determiner
        'SCONJ',  # subordinating conjunction
        'PUNCT',  # punctuation
        'SYM',  # symbol
        'X',  # other
        )
    ent_type = ent.ent_type_  # get entity type
    if ent_type == '':
        ent_type = 'NOUN_CHUNK'
        ent = ' '.join(str(t.text) for t in
                nlp(str(ent)) if t.pos_
                not in unwanted_tokens and t.is_stop == False)
    elif ent_type in ('NOMINAL', 'CARDINAL', 'ORDINAL') and str(ent).find(' ') == -1:
        t = ''
        for i in range(len(sent) - ent.i):
            if ent.nbor(i).pos_ not in ('VERB', 'PUNCT'):
                t += ' ' + str(ent.nbor(i))
            else:
                ent = t.strip()
                break
    return ent, ent_type


# In[5]:


def get_relation(sent):
    doc = nlp(sent)

    # Matcher class object 
    matcher = Matcher(nlp.vocab)

    #define the pattern 
    pattern = [{'DEP':'ROOT'}, 
              {'DEP':'prep','OP':"?"},
              {'DEP':'agent','OP':"?"},  
              {'POS':'ADJ','OP':"?"}] 

    matcher.add("matching_1", None, pattern) 
    matches = matcher(doc)
    k = len(matches) - 1

    span = doc[matches[k][1]:matches[k][2]] 

    return(span.text)


# In[6]:


def entity_pairs(text, coref=True):
    
    prv_tok_dep = ""    # dependency tag of previous token in the sentence
    prv_tok_text = ""   # previous token in the sentence

    prefix = ""
    modifier = ""
    
    
    text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
    text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
    text = nlp(text)
    
    if coref:
        text = nlp(text._.coref_resolved)  # resolve coreference clusters
    
    sentences = [sent.string.strip() for sent in text.sents]  # split text into sentences
    ent_pairs = list()
    
    for sent in sentences:
        sent = nlp(sent)
        spans = list(sent.ents) + list(sent.noun_chunks)  # collect nodes
        spans = spacy.util.filter_spans(spans)
        with sent.retokenize() as retokenizer:
            [retokenizer.merge(span) for span in spans]
        dep = [token.dep_ for token in sent]
        if (dep.count('obj')+dep.count('dobj'))==1                 and (dep.count('subj')+dep.count('nsubj'))==1:
            # Making sure the sentence only has a single object and single subject
            for token in sent:
                
                # check: token is a compound word or not
                if token.dep_ == "compound":
                    prefix = token.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        prefix = prv_tok_text + " "+ token.text
                
                
                # check: token is a modifier or not
                if token.dep_.endswith("mod") == True:
                    modifier = token.text
                    # if the previous word was also a 'compound' then add the current word to it
                    if prv_tok_dep == "compound":
                        modifier = prv_tok_text + " "+ token.text
                
                if token.dep_.find("subj") == True:
                    ent1 = modifier +" "+ prefix + " "+ token.text
                    prefix = ""
                    modifier = ""
                    prv_tok_dep = ""
                    prv_tok_text = ""      

                if token.dep_.find("obj") == True:
#                     if prv_tok_dep == "compound":
                    ent2 = prefix +" "+ token.text
                
                # update variables
                prv_tok_dep = token.dep_
                prv_tok_text = token.text
            
            # subj : ent1, obj : ent2
            # refine later
            subject = ent1.strip()
            token = ent2.strip()
            relation = get_relation(sent.text)
            ent_pairs.append([str(subject), str(relation), str(token)])
    
    filtered_ent_pairs = [sublist for sublist in ent_pairs
                          if not any(str(x) == '' for x in sublist)]    
    pairs = pd.DataFrame(filtered_ent_pairs, columns=['subject',
                         'relation', 'object'])
    print('Entity pairs extracted:', str(len(filtered_ent_pairs)))
    return pairs
    


# In[7]:


entity_pairs("""Abbu gives me some milk
and chochwor bread. The bread is
from Wasim’s family bakery. It has
a special touch: raisins. I love it.""")


# In[28]:


from nltk.tokenize import sent_tokenize
text = """The guidelines include a prohibition on the localized data collection system and limitations on the use of API which allows only one 
app per country to use the API.Over 1,000 migrant workers arrived in Lucknow today morning in a special train from Maharashtra's Akola district where they were stranded due to the coronavirus lockdown.
Four people, who recently returned from Gujarat and West Bengal, tested positive for the novel coronavirus in Odisha on Tuesday, taking the number of cases in the state to 173, an official said."""
text = re.sub(r'\n+', '.', text)  # replace multiple newlines with period
text = re.sub(r'\[\d+\]', ' ', text)  # remove reference numbers
print(sent_tokenize(text))


# In[ ]:


"""The guidelines include a prohibition on the localized data collection system and limitations on the use of API which allows only one 
app per country to use the API.Over 1,000 migrant workers arrived in Lucknow today morning in a special train from Maharashtra's Akola district where they were stranded due to the coronavirus lockdown.
Four people, who recently returned from Gujarat and West Bengal, tested positive for the novel coronavirus in Odisha on Tuesday, taking the number of cases in the state to 173, an official said."""


# In[ ]:


"""Global efforts to develop a vaccine against the coronavirus disease (Covid-19) have progressed at an unprecedented pace aiming to stop the spread of the pandemic, which has infected 3.5 million people, killed 250,000 and wrecked global economies within four months. 
At least 120 vaccine projects are in various stages of development since China shared the genetic sequence of Sars-CoV-2, which causes Covid-19, with the World Health Organization (WHO) on January, 12, 2020.
Pune-based Serum Institute of India started work 10 days ago on manufacturing in parallel to the human safety trials, the Oxford experimental vaccine, ChAdOx1 nCoV-19, at its own risk.
Covid-19 vaccines use a wide variety of platforms and techniques to train the immune system to identify the Sars-CoV-2 virus and block or destroy it before it infects the body.
"""


# In[8]:


pairs = entity_pairs("""Abbu gives me some milk
and chochwor bread. The bread is
from Wasim’s family bakery. It has
a special touch: raisins. I love it.""")


# In[9]:


import networkx as nx
import matplotlib.pyplot as plt


def draw_kg(pairs):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
            create_using=nx.MultiDiGraph())
    node_deg = nx.degree(k_graph)
    layout = nx.spring_layout(k_graph, k=0.15, iterations=20)
    plt.figure(num=None, figsize=(120, 90), dpi=80)
    nx.draw_networkx(
        k_graph,
        node_size=[int(deg[1]) * 500 for deg in node_deg],
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white',
        )
    labels = dict(zip(list(zip(pairs.subject, pairs.object)),
                  pairs['relation'].tolist()))
    nx.draw_networkx_edge_labels(k_graph, pos=layout, edge_labels=labels,
                                 font_color='red')
    plt.axis('off')
    plt.show()


# In[11]:


draw_kg(pairs)


# In[12]:


def filter_graph(pairs, node):
    k_graph = nx.from_pandas_edgelist(pairs, 'subject', 'object',
            create_using=nx.MultiDiGraph())
    edges = nx.dfs_successors(k_graph, node)
    nodes = []
    for k, v in edges.items():
        nodes.extend([k])
        nodes.extend(v)
    subgraph = k_graph.subgraph(nodes)
    layout = (nx.random_layout(k_graph))
    nx.draw_networkx(
        subgraph,
        node_size=1000,
        arrowsize=20,
        linewidths=1.5,
        pos=layout,
        edge_color='red',
        edgecolors='black',
        node_color='white'
        )
    labels = dict(zip((list(zip(pairs.subject, pairs.object))),
                    pairs['relation'].tolist()))
    edges= tuple(subgraph.out_edges(data=False))
    sublabels ={k: labels[k] for k in edges}
    print(k_graph.out_edges(data=False))
    nx.draw_networkx_edge_labels(subgraph, pos=layout, edge_labels=sublabels,
                                font_color='red')
    plt.axis('off')
    plt.show()


# In[13]:


filter_graph(pairs, 'Wasim')


# In[ ]:




