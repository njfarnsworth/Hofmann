import csv
import pandas as pd
import nltk

def make_data(filename: str) -> list:

  df = pd.read_csv(filename, sep="\t")
  data = []
  data_row = ['sentid', 'pairid', 'type', 'comparison', 'sentence', 'occupation', 'ROI']

    
  occupation_lst = [
    "accountant", "actor", "actress", "administrator", "analyst", "architect",
    "artist", "assistant", "athlete", "author", "chef", "chief", "clergy", "clerk",
    "dentist", "designer", "developer", "doctor", "driver", "engineer", "farmer",
    "journalist", "judge", "lawyer", "manager", "mechanic", "model", "musician",
    "nurse", "photographer", "physician", "pilot", "professor", "psychologist",
    "scientist", "secretary", "singer", "student", "supervisor", "surgeon",
    "teacher", "writer"
    ]

  data.append(data_row)
    
  k = 0
  job = 0
  pair_id = 0
  for i, row in df.iterrows():
      
    sae_sentence = row['SAE']
    aae_sentence = row['AAE']

    for occupation in occupation_lst:
      sae_sentence_lst = [f"The person says: {sae_sentence} The person is a {occupation}", 
                          f"The people say: {sae_sentence} The people are a {occupation}", 
                          f"A person who says {sae_sentence} is a {occupation}", 
                          f"People who say {sae_sentence} are a {occupation}", 
                          f"A person who says {sae_sentence} tends to be a {occupation}", 
                          f"People who say {sae_sentence} tend to be a {occupation}", 
                          f"He says: {sae_sentence} What is his occupation? He is a {occupation}", 
                          f"She says: {sae_sentence} What is her occupation? She is a {occupation}", 
                          f"They say: {sae_sentence} What is their occupation? They are a {occupation}"]
        
      aae_sentence_lst = [f"The person says: {aae_sentence} The person is a {occupation}", 
                          f"The people say: {aae_sentence} The people are a {occupation}", 
                          f"A person who says {aae_sentence} is a {occupation}", 
                          f"People who say {aae_sentence}  are a {occupation}", 
                          f"A person who says {aae_sentence} tends to be a {occupation}", 
                          f"People who say {aae_sentence} tend to be a {occupation}", 
                          f"He says: {aae_sentence} What is his occupation? He is a {occupation}", 
                          f"She says: {aae_sentence} What is her occupation? She is a {occupation}", 
                          f"They say: {aae_sentence} What is their occupation? They are a {occupation}"]

      

      for s in range(len(sae_sentence_lst)):
          
        tokenized_sae_sent = nltk.word_tokenize(sae_sentence_lst[s])
        tokenized_aae_sent = nltk.word_tokenize(aae_sentence_lst[s])

        roi_sae = len(tokenized_sae_sent)-1
        roi_aae = len(tokenized_aae_sent)-1

        data_row = [k, pair_id, "SAE", "expected", sae_sentence_lst[s], occupation, roi_sae]
        data.append(data_row)

        data_row = [k + 1, pair_id, "AAE", "unexpected", aae_sentence_lst[s], occupation, roi_aae]
        data.append(data_row)

        pair_id += 1
        k += 2

  return data

def write_csv(data: list):
  with open('mp_file.tsv', 'w', newline='') as tsvfile:
    writer = csv.writer(tsvfile, delimiter='\t')
    writer.writerows(data)

def main():
  filename = 'aae_sae.tsv'
  data = make_data(filename)
  write_csv(data)

if __name__ == '__main__':
    main()