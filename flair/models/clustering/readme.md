Flair cluster implementation 
--------------

TODO:
- BIRCH
- EM Soft Clustering
- Evaluation for k Means
- Evaluation for EM
- Evaluation for BIRCH
- tutorial.md

Questions:
- aktuelle Stand
- Zeitplan vorstellen
- mini batching similarity for torch
- wie oft Clustering durchlaufen für Ergebnisse?  5x mal die Regel (mean, standrad deviation)

Traceback (most recent call last):
  File "C:/Users/Bomke/Desktop/Kroatien/Studienarbeit/flair-text-clustering/run_EM.py", line 11, in <module>
    embeddings = SentenceTransformerDocumentEmbeddings('bert-base-nli-mean-tokens')
  File "C:\Python38\lib\site-packages\flair\embeddings\document.py", line 661, in __init__
    self.model = SentenceTransformer(model)
UnboundLocalError: local variable 'SentenceTransformer' referenced before assignment