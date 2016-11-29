## We will be using LSTM in Keras for this problem
## Also used word embeddings  
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
## For RNNs, trying without stemming and removing stopwords
# from collections import Counter
# from nltk.stem.porter import PorterStemmer
# from nltk.corpus import stopwords
from string import punctuation
# Import data
df = pd.read_csv("../input/Combined_News_DJIA.csv")
# print df.head()
# print df.shape
# print df.iloc[0,3]
names = df.columns.values 
### Dividing the data into test and train by dates (as specified)
trainStartDate = '2008-08-08' 
trainEndDate = '2014-12-31'
testStartDate = '2015-01-02'
testEndDate = '2016-07-01'
train = df[(df[names[0]]>=trainStartDate) & (df[names[0]]<=trainEndDate)]
test = df[(df[names[0]]>=testStartDate) & (df[names[0]]<=testEndDate)]
## Converting train into train_x and train_y 
trainX = train.loc[:,names[2:len(names)-1]]
## Subset by indices
# trainX = train.loc[:,[2,3]]
## Subset by indices
trainY = train.iloc[:,1]
## Subset by name
# trainY = train['Label']
# Similarly for test data. testX contains only textual information
testX = test.loc[:,names[2:len(names)-1]]
testY = test.iloc[:,1]
# print trainY.value_counts()
# 1    873
# 0    738
# delete all the other variables
del(train,test,trainStartDate,trainEndDate,testStartDate,testEndDate)

def join_reviews(df):
# Function to join multiple columns in each row to form corpus for each day
	joinedReviews = df.apply(lambda x: ''.join(str(x.values)), axis=1)
	return joinedReviews
## Joined the reviews together for each day
## Should devise a mechansim to weight the reviews
joinedTrainX = join_reviews(trainX)
joinedTestX = join_reviews(testX)
def pre_process(text):
# Basic form of pre-processing. Mainly removing 'b' character 
# representing bytes till 467 row.   
# Also remove punctuations and convert to lower case
	process_text=text.replace('\n','').replace('"b','').replace("'b",'')
	for punc in list(punctuation):
		process_text=process_text.replace(punc,'').lower()
	# process_text = process_text.strip(" ")
	# Remove extra spaces
	process_text=re.sub(' +',' ',process_text)
	return process_text
joinedTrainX = joinedTrainX.apply(lambda x: pre_process(x))
joinedTestX = joinedTestX.apply(lambda x: pre_process(x))
# Keras tokenizer and sequence padder 
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# Define max top words in vocabulary
top_words = 10000
tokenizer = Tokenizer(nb_words=top_words)
tokenizer.fit_on_texts(joinedTrainX)
# word_index = tokenizer.word_index
# print('Found %s unique tokens.' % len(word_index))
## Fit tokenizer
def generate_sequence(text,MAX_SEQUENCE_LENGTH):
	sequence= tokenizer.texts_to_sequences(text)
	sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)
	return sequence
## Maximum sequence length
max_words = 500
sequencesTrainX = generate_sequence(joinedTrainX,max_words)
sequencesTestX = generate_sequence(joinedTestX,max_words)
# result = map(len, sequencesTrainX)
# print("Mean %.2f words (%f)" % (np.mean(result), np.std(result)))
## Building the model
## First we will use word embedding and then LSTM layer over it 
from keras.models import Sequential
from keras.layers import Dense,LSTM,Flatten
from keras.layers.embeddings import Embedding
# Seed for reproducability
seed = 1
np.random.seed(seed)
## Dimension of embedding layer
embedding_dim = 32
model = Sequential()
model.add(Embedding(top_words, embedding_dim, input_length=max_words,dropout=0.2))
# model.add(Convolution1D(nb_filter=32, filter_length=3, border_mode='same', activation='relu'))
# model.add(MaxPooling1D(pool_length=2))
model.add(LSTM(128,dropout_W=0.2,dropout_U=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
batch_size = 32
# model.fit(sequencesTrainX, trainY, batch_size=batch_size, nb_epoch=1,verbose=1)
# scores = model.evaluate(X_test, y_test, verbose=1)
## Fit the model
model.fit(sequencesTrainX, trainY, batch_size=batch_size, nb_epoch=5,
          validation_data=(sequencesTestX, testY),verbose=1)
# Ideally should have separate validation data          
score, acc = model.evaluate(sequencesTestX, testY,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# Ideally should use grid search 
# Hyperparameters
# Vocabulary = 5000
# Max sequence length = 500
# Embedding dimension = 128
# Embedding dropout = 0.2
# Lstm Neurons = 128
# Epoch =5

# loss: 0.6653 - acc: 0.6102 - val_loss: 0.6924 - val_acc: 0.5132
# Test score: 0.692432944106
# Test accuracy: 0.513227514804

# Vocabulary = 10000
# loss: 0.4408 - acc: 0.8535 - val_loss: 0.7931 - val_acc: 0.5344
# Test score: 0.793134343056
# Test accuracy: 0.534391534549

# Vocabulary = 20000

# loss: 0.4714 - acc: 0.8597 - val_loss: 0.7335 - val_acc: 0.5106
# Test score: 0.733532233528
# Test accuracy: 0.510582010267