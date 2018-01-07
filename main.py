from sklearn.feature_extraction.text import TfidfVectorizer #Import a vectorizer
from sklearn.svm import LinearSVC #Import Support Vector Machine
from sklearn. metrics import accuracy_score #Metric calculator to measure how well the classifier performs


with open("clickbait.txt") as file:
    lines = file.read().strip().split("\n")
    lines = [line.split("\t") for line in lines]
headlines,labels = zip(*lines)


#Looking at the data
print("headlines : ")
for hl in headlines[:5]:
	print(hl)

print("labels : ")
print(labels[:5])

print("Size of dataset : ")
print(len(headlines))


#Splitting our dataset into Training and Test sets (80-20)

train_headlines = headlines[:8000] #first 8000 examples
train_labels = labels[:8000]


test_headlines = headlines[8000:] #last 2000 examples
test_labels = labels[8000:]


#  Using Linear SVM classifier

vectorizer = TfidfVectorizer()
svm = LinearSVC()


# Training our classifier

#our vectorizer assumes that words found in our training set represent all of the vocabulary
train_vectors = vectorizer.fit_transform(train_headlines)

#only call transform (without fit) on test set - any word not in training set will be discarded
test_vectors  = vectorizer.transform(test_headlines)

#Train the classifier 
svm.fit(train_vectors,train_labels)


# ### Evaluating our Classifier

predictions = svm.predict(test_vectors)


print("Test headlines : ")
print(test_headlines[:5])

print("Predictions : ")
print(predictions[:5])

print("Test labels")
print(test_labels[:5])

print("Accuracy : ",accuracy_score(test_labels,predictions))

new_headlines = ["10 Cities That Every Hipster Will Be Moving To Soon", 'Vice President Mike Pence Leaves NFL Game Saying Players Showed "Disrespect" Of Anthem, Flag']
new_vectors = vectorizer.transform(new_headlines)
new_predictions = svm.predict(new_vectors)


print("New headlines : ")
print(new_headlines[0],new_predictions[0])
print(new_headlines[1],new_predictions[1])
