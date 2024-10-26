# IE691-BioASQ-13b

This repository contains the source code for our group's IE691 project. The project is on Biomedical Semantic QA, Task 13b.

Group Members:
1. Phelan Lee Yeuk Bun
2. Aron Andika
3. Ryu Kairin Anaqi Suzuki
4. Teo Zhuo Hang

Relevant Links:
http://participants-area.bioasq.org/general_information/Task13b/

## Data Acquisition
The data acquisition process entails retrieving relevant articles from the Pubmed database. The steps for this include:
1. Import the questions from the JSON file.
2. Based on the question given, preprocess the query and extract the keywords/terms.
3. Using NCBI's eutils API, retrieve all relevant document IDs and their abstracts from Pubmed (i.e. documents that contains the keywords from Step 1).

## Retrieve a list of at most 10 relevant articles (Phase A)
This process entails ranking the retrieved documents from the Data Acquisition process in terms of similarity to the question/query given. The steps for this include:
1. Preprocess the query and the document abstracts (i.e. stopwords removal, lemmatization, spelling correction, capitalisation standardisation, special characters removal).
2. Convert the query and document abstracts to vectors/embeddings.
3. Calculate the cosine similarity between the query and each document abstract and rank the articles based on decreasing cosine similarity.

## Retrieve a list of at most 10 relevant text snippets (Phase A)
This process entails ranking the sentences from the document's abstract in terms of most relevance to the question/query. The steps for this include:
1. Retrieving the abstracts of the documents from the previous process. These are the documents that have been identified as part of the 10 most relevant to the question/query.
2. Split the abstract into sentences.
3. Convert the query and document abstracts to vectors/embeddings.
4. Calculate the dot product between the query and each document and rank the articles based on decreasing dot score. (We use dot product for finding the most relevant text snippets as we want the snippets that contain the most amount of relevant information i.e. vectors of higher magnitude).

## TO-DO
1. Add logger
2. Phase B