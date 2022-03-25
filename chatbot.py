import random
import json
import pickle
import numpy as np
import sys, os

import os

from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

import nltk
from nltk.stem import WordNetLemmatizer

from autocorrect import Speller

##loading in data from model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

spell = Speller(lang='en')

##cleaning up sentences

def clean_up_sentence(sentence):
	sentence_words = nltk.word_tokenize(sentence)
	sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
	return sentence_words

##getting bag of words

def bag_of_words(sentence):
	sentence_words = clean_up_sentence(sentence)
	bag = [0] * len(words)
	for w in sentence_words:
		for i, word in enumerate(words):
			if word == w:
				bag[i] = 1
	return np.array(bag)

##predicting class

def predict_class(sentence):
	bow = bag_of_words(sentence)
	res = model.predict(np.array([bow]))[0]
	error_threshold = .03
	results = [[i, r] for i, r in enumerate(res) if r > error_threshold]

	results.sort(key=lambda x: x[1], reverse=True)
	return_list = []
	for r in results:
		return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
	return return_list

##getting responce

def get_response(intents_list, intents_json):
	tag = intents_list[0]['intent']
	list_of_intents = intents_json['intents']
	for i in list_of_intents:
		if i['tag'] == tag:
			follow_ups_ = i['followups']
			result = [(random.choice(i['responses0'])+random.choice(i['responses1'])+random.choice(i['responses2'])).format(name), follow_ups_]
			break
	return result

def get_all_responses(var_name):
	list0_, list1_ = [], intents['intents']
	for i in list1_:
		if i['tag'] == var_name:
			for s0_ in i['responses0']:
				for s1_ in i['responses1']:
					for s2_ in i['responses2']:
						list0_.append(s0_ + s1_ + s2_)
	return(list0_)
get_all_responses('training questions')

## variable finder

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

## getting all responses
    
def get_all_responses(var_name):
	list0_, list1_ = [], intents['intents']
	for i in list1_:
		if i['tag'] == var_name:
			for s0_ in i['responses0']:
				for s1_ in i['responses1']:
					for s2_ in i['responses2']:
						list0_.append(s0_ + s1_ + s2_)
	return(list0_)
        
## remove this when you are done bridging gate between app and server
name = "Marcos"

def chat(message):
	message = spell(message.lower())
	add_to_database_ = get_all_responses('add to database now')
	name = "Marcos"
	res = 'temp'
    ## bot training and storing after training
	if res == var_finder(add_to_database_, res, res, False):
		message_0 = input('').lower()
		list_temp = []
		list_temp.append(message)
		list_temp.append(message_0)
		res = get_response([{'intent': 'training done', 'probability': '1'}], intents)[0]
		followups = get_response([{'intent': 'training done', 'probability': '1'}], intents)[1]
		if followups == [""]:
			return (res, "")
		else:
			return (res, followups)

	else:
		ints = predict_class(message)
        ## bot is certain enough its the correct response 
		if float(ints[0]['probability']) > .8:
			res = get_response(ints, intents)[0]
			followups = get_response(ints, intents)[1]
        ## bot is uncertain answer is correct result
		else:
			res = get_response([{'intent': 'bot uncertain', 'probability': '1'}], intents)[0]
			followups = get_response([{'intent': 'bot uncertain', 'probability': '1'}], intents)[1]
		if followups == [""]:
			return (res, "")
		else:
			return (res, followups)


if __name__ == "__main__":
	os.system('cls')
	print('Bot is on, say hi!')
	while True:
		message = input("You: ")
		if message == "break":
			print('turning off')
			break
		print(chat(message))
