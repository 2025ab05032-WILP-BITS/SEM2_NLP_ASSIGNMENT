Natural Language Processing S2-25_AIMLCZG530 Assignment 1 Problem Statement - 12 

**General Instructions** 

1. Implementation on the BITS OSHA Virtual Lab is compulsory, and carries 1 mark. 


2. Submit the code in .pdf/html format along with output. Please follow the naming convention as .pdf. Example for group 1 should be named as - Group1.pdf. 


3. Include group details like team members' names, BITS ID, and contribution percentages. 


4. All members of the group will work on the same problem statement. Each group should upload in Taxila in respective locations under ASSIGNMENT Tab. Assignments submitted via means other than through Taxila will not be graded. 


5. One student per group is requested to submit your assignment. 


6. Any queries related to this problem statement should be addressed to Vasugi I (vasugii@wilp.bits-pilani.ac.in), Course LF. 


7. If the dataset link is not working, download a similar dataset from an online resource and note this in your submission. 



**Feed-Forward Neural Language Model** Link to the Dataset: [https://www.kaggle.com/datasets/ilhamferdiona/ag-news-classification-dataset](https://www.kaggle.com/datasets/ilhamferdiona/ag-news-classification-dataset) Description of Data: AG News dataset with 120,000 news articles from 4 categories (World, Sports, Business, Sci/Tech). Use the first 5,000 articles. 

1. Download the file and set it as a DataFrame. Use the 'Text' column only. (1 Mark) 


2. Remove punctuations, numbers, and stopwords. Convert to lowercase and tokenize. Create vocabulary with minimum frequency = 3 (2 Marks) 


3. Create input-output pairs with context window = 3 words. Design neural network:  Embedding(300d) → Hidden(128) → ReLU → Output(Vocab_size). Display embedding for the second most frequent word. (3 Marks) 


4. Train model for 10 epochs with cross-entropy loss. Plot training loss curve and display final loss value. (3 Marks) 


5. Compare and contrast Count-based (N-Gram) vs Prediction-based (Neural LM)  embedding techniques in capturing semantic relationships. Which performs better for rare words? (2 Marks) 


