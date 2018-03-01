

text = 'This is a random text just for practice and i should write some more like this.'

def count_words(text):
    """
    This function just count the number of words in a string of words. This returns
    a dictionary with values as no of counts of words. Removes punctuation.
    """
    text = text.lower()
    word_count = {}
    skips = [".",",",";",":","'",'"']
    for ch in skips:
        text = text.replace(ch,'')
    for word in text.split(' '):
        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
    return word_count
    

from collections import Counter

def count_words_fast(text):
    """
    This function just count the number of words in a string of words. This returns
    a dictionary with values as no of counts of words. Removes punctuation.
    """
    text = text.lower()
    word_count = {}
    skips = [".",",",";",":","'",'"']
    for ch in skips:
        text = text.replace(ch,'')
    word_count = Counter(text.split(' '))
    return word_count

def read_book(title_path):
    with open(title_path,'r',encoding = 'utf8') as current_file:
        text = current_file.read()
        text = text.replace('\n','').replace('\r','')
    return text


def word_stats(word_counts):
    """ Return count of unique words and their frequency """
    unique_words = len(word_counts)
    freq = word_counts.values()
    return(unique_words,freq)

text = read_book("Romeo and Juliet.txt")
words = count_words(text)
(unique_words,freq) = word_stats(words)
print(unique_words,sum(freq))

text = read_book("Romeo und Julia.txt")
words = count_words(text)
(unique_words,freq) = word_stats(words)
print(unique_words,sum(freq))


# Reading multiple books

import os
book_dir = ".\Python_case_study\Books"

import pandas as pd

stats = pd.DataFrame(columns = ("Language","Author","Title","Unique","Count"))
title_num = 1

for language in os.listdir(book_dir):
    for author in os.listdir(book_dir + "/" + language):
        for title in os.listdir(book_dir + "/" + language + "/" + author):
            input_file = book_dir + "/" + language + "/" + author + "/" + title
            print(input_file)
            text = read_book(input_file)
            (unique_words,count) = word_stats(count_words(text))
            stats.loc[title_num] = language,author.capitalize(),title.replace(".txt",""),unique_words,sum(count)
            title_num += 1
            

# Plotting the data obtained from reading the books
            

import matplotlib.pyplot as plt

plt.plot(stats.Count,stats.Unique,"rs")
plt.loglog(stats.Count,stats.Unique,"ro")


plt.figure(figsize = (10,10))
subset = stats[stats.Language == "English"]
plt.loglog(subset.Count,subset.Unique,"o",label = "English",color = "crimson")
subset = stats[stats.Language == "French"]
plt.loglog(subset.Count,subset.Unique,"o",label = "French",color = "orange")
subset = stats[stats.Language == "German"]
plt.loglog(subset.Count,subset.Unique,"o",label = "German",color = "forestgreen")
subset = stats[stats.Language == "Portuguese"]
plt.loglog(subset.Count,subset.Unique,"o",label = "Portuguese",color = "blueviolet")
plt.legend()
plt.xlabel('Number of Words in the book')
plt.ylabel('Number of unique words in the book')
plt.savefig('lang_fig.pdf')
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            


            
































































