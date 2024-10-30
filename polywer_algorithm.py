import numpy as np
from jiwer import cer
from transformers import AutoTokenizer, AutoModel
import torch, re

def cleanList(l):
    l = l.lower()
    l = re.sub(r"[^a-zA-Z0-9\s\[\]\u0621-\u063A\u0641-\u064A\u0660-\u0669]", '', l) # Removing characters that aren't latin/arabic letters/square brackets
    l = re.sub("[إأٱآا]", "ا", l) # Normalizing alef: إ أ ٱ آ ا -> ا
    l = re.sub("[ةه]", "ه", l) # Normalizing ta2 marbuta: ةه -> ه
    l = re.sub(r'([ىيئ])(?=\s|[.,!?؛:\]،]|$)', 'ى', l) # Normalizing ya2 when it occurs at the end of the word: ى ي ئ -> ى
    l = l.replace(' ]', ']') # Stripping spaces inside square brackets
    l = l.replace('[ ', '[')
    l = l.replace(']', '] ') # Adding spaces outside square brackets
    l = l.replace('[', ' [')
    l = l.strip() # Removing double spaces
    l = l.split()
    return l

def alignTranslation(r):
    '''
        This function processes a list `r` where:
        - `r[0]` contains the original code-switched (cs) transcription.
        - `r[1]` contains a version of the cs transcription with transliterations.
        - `r[2]` contains a version of the cs transcription with translations with possible length mismatches.

        Given:
        - `n`: the number of words between two brackets in the cs transcription.
        - `m`: the number of words between two brackets in the translated transcription.

        The function groups the `m` words between brackets in the translated transcription and duplicates them `n` times to match the length.

        Once aligned, for any index `i`, `r[0][i]` will correspond to `r[3][i]`.
        
        Example
        Input:
            `r[0]`:   ['أقدر', 'أفتح', '[coffee', 'shop]', 'عشان', '[I', 'put]', 'كل', 'اللي', 'عندي']    len: 10
            `r[1]`:   ['أقدر', 'أفتح', '[كوفي', 'شاب]', 'عشان', '[آي', 'بوت]', 'كل', 'اللي', 'عندي']      len: 10
            `r[2]`:   ['أقدر', 'أفتح', '[مقهى]', 'عشان', '[أعطي]', 'كل', 'اللي', 'عندي']                  len: 8

        Output:
            `r[0]`:   ['أقدر', 'أفتح', '[coffee', 'shop]', 'عشان', '[I', 'put]', 'كل', 'اللي', 'عندي']        len: 10
            `r[1]`:   ['أقدر', 'أفتح', '[كوفي', 'شاب]', 'عشان', '[آي', 'بوت]', 'كل', 'اللي', 'عندي']          len: 10
            `r[2]`:   ['أقدر', 'أفتح', '[مقهى]', 'عشان', '[أعطي]', 'كل', 'اللي', 'عندي']                      len: 8
            `r[3]`:   ['أقدر', 'أفتح', '[مقهى]', '[مقهى]', 'عشان', '[أعطي]', '[أعطي]', 'كل', 'اللي', 'عندي']  len: 10

        
        Parameters:
        `r` (list of lists): The input list containing code-switched, transliterated and translated transcriptions.
        
        Returns:
        `r` (list of lists): The modified list `r` with the aligned translated transcription.


    '''

    r.append([])        # Initializing the new list for the aligned translation

    in_brackets = False # Flag for CS regions
    count = 0           # Counter for n, the number of words between 2 brackets in the cs transcription
    section = ''        # The words between brackets in the translated transcription to be grouped and duplicated
    j = 0               # Pointer for the translation transcription

    for i in range(len(r[0])):      # Pointer for the cs transcription
        if r[0][i][0] == '[':
            in_brackets = True 
            if j < len(r[2]):
                section += r[2][j]   
                
            while j + 1 < len(r[2]) and r[2][j][-1] != ']': # Collecting all the CS words in the translation
                j += 1
                section += ' ' + r[2][j]

        if (in_brackets):  
            count += 1  
            if(r[0][i][-1] == ']'):     # Checking if we're done with the current brackets in the original cs transcription
                for num in range(count):
                    r[3].append(section)    # Adding m translated words n times to the aligned translation

                j += 1
                count = 0
                in_brackets = False
                section = ''
        else:
            if j < len(r[2]):
                r[3].append(r[2][j])    # Adding a regular word (not between brackets) to the aligned translation
                j += 1

    return r


def computeCosine(tokenizer, model, r_word, h_word):
    '''
    This function computes the cosine similarity between the BERT embeddings of two words.  
    
    Parameters:
        `tokenizer` (AutoTokenizer): The BERT tokenizer for tokenizing the input words.
        `model` (AutoModel): The pre-trained BERT model for obtaining the embeddings.
        `r_word` (str): The reference word to compare.
        `h_word` (str): The hypothesis word to compare.  

    Returns:
        `cosine` (float): The cosine similarity between the embeddings of the two words.
    '''

    r_tok = tokenizer(r_word, return_tensors="pt")
    h_tok = tokenizer(h_word, return_tensors="pt")
    r_emb = model(**r_tok).last_hidden_state.mean(dim=1).squeeze()
    h_emb = model(**h_tok).last_hidden_state.mean(dim=1).squeeze()
    cosine = torch.nn.functional.cosine_similarity(r_emb.unsqueeze(0), h_emb.unsqueeze(0)).item()
    return cosine


def editDistance(r, h, cer_threshold, cosine_threshold):
    '''
    This function calculates the edit distance of the reference and the hypothesis using dynamic programming.

    Parameters: 
        `r`: a list of lists of transcriptions:
          `r[0]`: list of words produced by splitting original reference (CS).
          `r[1]`: list of words produced by splitting transliterated sentece.
          `r[2]`: list of words produced by splitting translated sentece. 
        `h`: a list of words produced by splitting hypothesis sentence.
        `cer_threshold` (float): maximum value for CER between hypothesis and transliterated words
        `cosine_threshold` (float): minimum value for cosine similarity between hypothesis and translated words

    Returns:
        `d` (list of lists): the matrix representing the computed edit distances between the reference and hypothesis
    
    '''

    # Initializind d
    d = np.zeros((len(r[0])+1)*(len(h)+1), dtype=np.float64).reshape((len(r[0])+1, len(h)+1))
    tokenizer = AutoTokenizer.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-da')
    model = AutoModel.from_pretrained('CAMeL-Lab/bert-base-arabic-camelbert-da')
    # cer_threshold = .25
    # cosine_threshold = .85

    # Initializing d matrix
    for i in range(len(r[0])+1):
        d[i][0] = i
    for j in range(len(h)+1): 
        d[0][j] = j

    # Filling d matrix
    for i in range(1, len(r[0])+1):
        for j in range(1, len(h)+1):
            if r[0][i-1].strip('[]') == h[j-1]:             # Comparing hypothesis with cs transcription
                d[i][j] = d[i-1][j-1]

            else: 
                # Calculating CER score between hypothesis and transliterated words
                if r[1][i-1] == r[0][i-1]: translit = d[i-1][j-1] + 1 # Not a CS region (OG[word] == LIT[word]) -> treat as substitution
                else:
                    cer_score = cer(r[1][i-1].strip('[]'), h[j-1]) 
                    cer_score = round(cer_score, 2)
                    translit = min(d[i-1][j-1], d[i][j-1], d[i-1][j]) + cer_score if cer_score <= cer_threshold else d[i-1][j-1] + 1

                # Calculating cosine similarity score between hypothesis and translated words
                if r[3][i-1] == r[0][i-1] or cosine_threshold == -1: translat = d[i-1][j-1] + 1 # Not a CS region (OG[word] == LAT[word]) -> treat as substitution
                else:
                    cosine = 0
                    for word in r[3][i-1].split():              # Going over each word in translated section
                        word = word.strip('[]')
                        new_cosine = round(computeCosine(tokenizer, model, word, h[j-1]),2)
                        new_cosine = np.round(new_cosine, 2)
                        cosine = max(cosine, new_cosine)        # Keeping the highest cos sim value 
                    translat = min(d[i-1][j-1], d[i][j-1], d[i-1][j]) + 1 - cosine if cosine > cosine_threshold else d[i-1][j-1] + 1
    
                substitute = d[i-1][j-1] + 1
                insert = d[i][j-1] + 1
                delete = d[i-1][j] + 1
                
                d[i][j] = min(substitute, insert, delete, translit, translat)   # Selecting the lowest score

    return d


def polywer(r, h, cer_threshold, cosine_threshold):

    r[0] = cleanList(r[0])
    r[1] = cleanList(r[1])
    r[2] = cleanList(r[2])
    h = cleanList(h)

    r = alignTranslation(r.copy()) 
    
    assert len(r[0]) == len(r[1]) and len(r[0]) == len(r[3])

    d = editDistance(r, h, cer_threshold, cosine_threshold)

    result = float(d[len(r[0])][len(h)]) / len(r[0])

    return result


def polywer_multi(r_list, h_list, cer_threshold, cosine_threshold):
    num_errors = []
    num_words = []

    for r, h in zip(r_list, h_list):
        r[0] = cleanList(r[0])
        r[1] = cleanList(r[1])
        r[2] = cleanList(r[2])
        h = cleanList(h)
        
        r = alignTranslation(r.copy())
        
        assert len(r[0]) == len(r[3]) and len(r[0]) == len(r[1])

        d = editDistance(r, h, cer_threshold, cosine_threshold)
        num_errors.append(d[-1][-1])
        num_words.append(len(r[0]))
    
    total_errors = sum(num_errors)
    total_words = sum(num_words)

    return total_errors/total_words
