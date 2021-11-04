import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model
from autocorrect import Speller

## loading in data from model and intializing models and libs

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')
spell = Speller(lang='en')

## cleaning up sentences

def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
	return sentence_words

## getting bag of words

def bag_of_words(sentence):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)
	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)

## predicting class

def predict_class(sentence):
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]
	error_threshold = .25
	results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
	return return_list

## getting responce

def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if i['tag'] == tag:
			result = random.choice(i['responses'])
			break
	return result


##functions to sort through responses, and refrence if response is in list

def var_finder(lst, var_searching, return_var, fail_var):
    var_ = 0
    for i in lst:
        if i == var_searching:
            var_ = 1
            break
    if var_ == 1:
        return (return_var)
    else:
        return (fail_var)
    
def get_all_responses(var_name):
    list_ = intents['intents']
    for i in list_:
        if i['tag'] == var_name:
            return((i['responses']))

print('Bot is on, say hi!')


res = 'temp'

while True:
	add_to_database = get_all_responses('add to databse now')
	message = spell(input('').lower())

	## test variable, remove after training is done
	if message == 'break':
		print('turning off')
		break

	elif res == var_finder(add_to_database, res, res, False):
		message_0 = input('').lower()
		list_temp = []
		list_temp.append(message)
		list_temp.append(message_0)
		res = get_response([{'intent': 'training done', 'probability': '1'}], intents)
		print(res)


	else:
		ints = predict_class(message)
        ## bot is certain enough its the correct response 
		if float(ints[0]['probability']) > .99:
			res = get_response(ints, intents)
        ## bot is uncertain answer is correct result
		else:
			res = get_response([{'intent': 'bot uncertain', 'probability': '1'}], intents)
		print(res)