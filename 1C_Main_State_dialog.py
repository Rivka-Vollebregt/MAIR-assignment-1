### Helper functions
# Create neural network model
# Import library's
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential 
from keras import layers
from keras.backend import clear_session

#Import data file
df = pd.read_csv(('dialog_acts.dat'),names=['label'])

# Transform to lowercase
df['label'] = df['label'].str.lower()
df[['label','utterance']] = df['label'].str.split(" ",1,expand=True)

# split into 85% train and 15% test data
mask = np.random.rand(len(df)) <= 0.85
training_data = df[mask]
testing_data = df[~mask]

# Copy train and test data for this model
m1_sentences_train = training_data['utterance']
m1_sentences_test = testing_data['utterance']
m1_y_train = training_data['label']
m1_y_test = testing_data['label']

# Vectorize the training and testing x data
m1_vectorizer = CountVectorizer()
m1_vectorizer.fit(m1_sentences_train)

m1_x_train = m1_vectorizer.transform(m1_sentences_train)
m1_x_test = m1_vectorizer.transform(m1_sentences_test)

# Categorize the labels (y data)
label_encoder = LabelEncoder()
m1_y_train = label_encoder.fit_transform(m1_y_train)
m1_y_test = label_encoder.fit_transform(m1_y_test)

m1_y_train = keras.utils.to_categorical(m1_y_train, 18)
m1_y_test = keras.utils.to_categorical(m1_y_test, 18)

m1_input_dim = m1_x_train.shape[1]

# Architecture - layers
model1 = Sequential()
model1.add(layers.Dense(28, input_dim=m1_input_dim, activation = 'relu'))
model1.add(layers.Dense(18, activation='sigmoid'))
model1.compile(loss='binary_crossentropy', optimizer='adam', metrics='accuracy')

# Train model and print training epochs
history = model1.fit(m1_x_train, m1_y_train, epochs=15, verbose=0 , validation_data=(m1_x_test, m1_y_test), batch_size=100)


# All possible response types (15 types)
list_labels = ["ack","affirm","negate","inform","thankyou","bye", "restart","request","reqmore",  "reqalts", "repeat", "hello" ,"deny","confirm", "null"]

# Function to use for predicting the dialog_act
def pred_dialog(pred_sentence):
  pred_sentence = m1_vectorizer.transform([pred_sentence])
  y_new = model1.predict([pred_sentence])
  y_class = y_new.argmax(axis=-1)
  return sorted(list_labels)[y_class[0]]

# Determines levenshteinDistance between two elements
# Calculate likeness of two words with levenshtein distance
def levenshteinDistanceMatrix(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))
    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1
    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2
        
    a = 0
    b = 0
    c = 0
    
    # Calculate distance between token 1 and 2
    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]
                
                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]
  
  # Find the words that are closest to the input word
def closest_area(sentence,dictionary):

  # Will store all words and distances if < 3
  dictWordDist = []

  # Go through every word in sentence
  for word in sentence:
    if len(word) > 3:
      wordIdx = 0
        
      # Check the word against every dictionary element
      for line in dictionary: 
          wordDistance = levenshteinDistanceMatrix(word, line.strip())
          if wordDistance >= 10:
              wordDistance = 9
          # Only include words with distance less then three
          if wordDistance < 3:
            dictWordDist.append(str(int(wordDistance)) + "-" + line.strip())
            wordIdx = wordIdx + 1

  # Go through all distances found for every word in sentence
  closestWords = []

  # If there are suitable words found, sort and find the closest word
  if len(dictWordDist) > 0:
    wordDetails = []
    currWordDist = 0

    # Sort list
    dictWordDist.sort()

    # Select shortest distance match
    currWordDist = dictWordDist[0]
    wordDetails = currWordDist.split("-")
    closestWords.append(wordDetails[1])
    
    # For converting list element to string, make empty string and join element in
    return_str = ""
    return return_str.join(closestWords) + " ?"
  else:
    return ""
  
  # Function to find keyword user preferences for area, food and price in string
# Utterance has to be a list of string words
def keyword(utterance):
  no_preference = ["any","don't care", "doesn't matter"]

  # Find area
  area = []
  
  # Possible areas
  dictionary_area = ["north","south","east","west","centre"]

  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_area:
      area = word
      break
    else:
      if word in no_preference:
        food_type = "*"
        break
  # Check if there's a word close to the input word
  if len(area) < 1:
      area = closest_area(utterance,dictionary_area)

  # Find food type
  food_type = []

  # Possible food types
  dictionary_food = ["british","european","italian","romanian","seafood","chinese",
                "steakhouse","asian","french","portugese","indian","spanish"
                "vietnamese","korean","thai","moroccan","swiss","fusion",
                "gastropub","tuscan","international","traditional","mediterranean","polynesian"
                "african","turkish","bistro","american","australasian","australasian",
                "persian","jamaican","lebanese","cuban","japanese","catalan"]
                
  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_food:
      food_type = word
      break
    else:
      if word in no_preference:
        food_type = "*"
        break
  # Check if there's a word close to the input word
  if len(food_type) < 1:
      food_type = closest_area(utterance,dictionary_food)

  # Find price
  price = []

  # Possible price ranges
  dictionary_price = ["moderate","expensive","cheap"]

  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_price:
      price = word
      break
    else:
      if word in no_preference:
        price = "*"
        break

  # Check if there's a word close to the input word
  if len(price) < 1:
      price = closest_area(utterance,dictionary_price)

  # Return found user preferences
  return area, food_type, price

def additional_requirements(utterance):
  food_quality = []
  crowdedness = []
  length_stay = []

  # Find food_quality
  dictionary_food_quality = ["good","bad"]
  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_food_quality:
      break
  # Check if there's a word close to the input word
  if len(area) < 1:
      food_quality = closest_area(utterance,dictionary_area)

  # Find crowdedness
  dictionary_crowdedness = ["busy","not busy"]
  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_crowdedness:
      break
  # Check if there's a word close to the input word
  if len(area) < 1:
      crowdedness = closest_area(utterance,dictionary_area)

  # Find crowdedness
  dictionary_length_stay = ["short","long"]
  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_length_stay:
      break
  # Check if there's a word close to the input word
  if len(area) < 1:
      length_stay = closest_area(utterance,dictionary_area)

  return food_quality, crowdedness, length_stay

# Recommnedation lookup function
import pandas as pd
import numpy as np
import random

# Find a matching recommendation from database restaurant_info.csv
def get_recommendations(area, food_type, price, recommendations):
  df = pd.read_csv("restaurant_info.csv") 
  # If area, food or pricerange is any, don't include in search
  if area=='any':
    recommendations=df[ (df.food==food_type) & (df.pricerange==price)]
  elif food_type =='any':
      recommendations=df[(df.area == area)  & (df.pricerange==price)]
  elif price=='any':
    recommendations=df[(df.area == area) & (df.food==food_type)]
  elif price=='any' and area=='any' :
    recommendations=df[ (df.food==food_type)]
  elif price=='any' and food_type =='any' :
    recommendations=df[ (df.area == area)]  
  elif area =='any' and food_type=='any':
    recommendations=df[(df.pricerange==price)]
  else:  
    recommendations=df[(df.area == area) & (df.food==food_type) & (df.pricerange==price)]
  return recommendations

#Option lookup function
import pandas as pd
import numpy as np
import random
#Find a matching options from recommended restaurants
def get_options(food_quality,crowdedness,length_stay,recommendations):
  if len(food_quality) < 1:
      recommendations=recommendations[ (recommendations.crowdedness==crowdedness) & (recommendations.lengthofstay==length_stay)]
  elif len(crowdedness) < 1:
      recommendations=recommendations[(recommendations.foodquality == food_quality)& (recommendations.lengthofstay==length_stay)]
  elif len(length_stay) < 1:
      recommendations=recommendations[(recommendations.foodquality == food_quality) & (recommendations.crowdedness==crowdedness)]
  elif len(food_quality) < 1 and len(crowdedness) < 1:
      recommendations=recommendations[(recommendations.lengthofstay==length_stay)]
  elif len(food_quality) < 1 and len(length_stay) < 1:
      recommendations=recommendations[(recommendations.crowdedness==crowdedness)]     
  elif len(crowdedness) < 1 and len(length_stay) < 1:
      recommendations=recommendations[(recommendations.foodquality==food_quality)]
  else:  
    recommendations=recommendations[(recommendations.foodquality == food_quality) & (recommendations.crowdedness==crowdedness) & (recommendations.lengthofstay==length_stay)]
  return recommendations

# Function to find type of dialog in a string based on keyword
def dialog_type(utterance):

  # Check the possible dialog_type keywords
  if utterance.find("okay") > 0 or utterance.find("oke") > 0:
    keyword = "ack"
  elif utterance.find("yes") > 0 or utterance.find("right") > 0:
    keyword = "affirm"
  elif utterance == "no":
    keyword = "negate"
  elif utterance.find("how about") > 0 or utterance.find("what about") > 0 or utterance.find("else") > 0:
    keyword = "reqalts"
  elif utterance.find("thank you") > 0 or utterance.find("thankyou") > 0 or utterance.find("thanks") > 0:
    keyword = "thankyou"
  elif utterance.find("bye") > 0:
    keyword = "bye"
  elif utterance.find("start") > 0:
    keyword = "restart"
  elif utterance.find("what") > 0 or utterance.find("how") > 0 or utterance.find("address") > 0 or utterance.find("number") > 0 or utterance.find("postcode") > 0:
    keyword = "request"
  elif utterance.find("more") > 0:
    keyword = "reqmore"
  elif utterance.find("repeat") > 0:
    keyword = "repeat"
  elif utterance.find("hi") > 0 or utterance.find("hello") > 0:
    keyword = "hello" 
  elif utterance.find("don't") > 0 or utterance.find("dont") > 0:
    keyword = "deny"
  elif utterance.find("is it") > 0 or utterance == "is":
    keyword = "confirm"
  elif utterance.find("looking") > 0 or utterance.find("want") > 0 or utterance.find("would") > 0 or utterance.find("any") > 0 or utterance.find("restaurant") > 0 or utterance.find("food") > 0:
    keyword = "inform"
  elif len(utterance) < 7 or utterance.find("noise") > 0 or utterance.find('sil') > 0:
    keyword = "null"
  else:
    keyword = "inform"

  return keyword

### Main: States
# 1 preference request
# 2 ask for missing preferences
# 3 ask for additional requirements and adjust restaurant options
# 4 recommend restaurant based on preferences
# 5 listen for information requests

# As extra feature for part two we add a delay before showing responses
# Amount of delay time can be edited below (seconds)
import time
delay = 0

# Extra feature: informal or formal 
formal = 0

# Dialog system
state = 1
dialog_active = True
need_ack = False

# Initiate preferences
area = ""
food_type = ""
price = ""

recommendations = []
index = 0

# Until user ends conversation, check for input and handle accordingly
while dialog_active:
  if state == 1:

  # Start the conversation with the welcome sentence
    if formal:
      print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")
    else:
      print("Hey, I can help recommend a restaurant. What kind of area, price range and food do you want?")
  # Go to next state after asking the question
    state = 2

  # State 2 is the state to get the missing preferences and allows the user to update previous preferences
  if state == 2:
    
      print("pref: ", area, ",", food_type,",", price)
      # Get user input for missing preferences
      user_input = input()
      user_input = user_input.lower()

      if need_ack:
        if pred_dialog(user_input.lower()) == "ack" or pred_dialog(user_input.lower()) == "affirm":
          if new_area != "":
            area = new_area
          if new_food_type != "":
            food_type = new_food_type
          if new_price != "":
            price = new_price
          need_ack = False
        else:
          need_ack = False

      new_area, new_food_type, new_price = keyword(user_input.split())


#     print(keyword(user_input.split()))
      # Area retrieval code
      if new_area != "":
        if new_area.find("?") != -1:
          new_area = new_area.replace(" ?", "")
          print("Did you say you are looking for a restaurant in the %s of town?" %new_area)
          need_ack = True
        elif new_area == "*":
          print("Let me confirm, you are looking for a restaurant in any area of town?")
          need_ack = True
        elif new_area.find("?") == -1:
          area = new_area

      # Food type retrieval code
      if new_food_type != "":
        if new_food_type.find("?") != -1:
          new_food_type = new_food_type.replace(" ?", "")
          print("You are looking for a %s restaurant right?" %new_food_type)
          need_ack = True
        elif new_food_type == '*':
          print("You are looking for a restaurant and do not care about the food right?")
          need_ack = True
        elif new_food_type.find("?") == -1:
          food_type = new_food_type

      # Price retrieval code
      if new_price != "":
        if new_price.find("?") != -1:
          new_price = new_price.replace(" ?", "")
          print("Let me confirm, you are looking for a restaurant in the %s price range?" %new_price)
          need_ack = True
        elif new_price.find("*") != -1:
          print("Let me confirm, you are looking for a restaurant in any price range?")
          need_ack = True
        elif new_price.find("?") == -1:
          price = new_price

      # Update old preferences when new preference != ""
      if need_ack == False:
        if new_area != "":
          area = new_area
        if new_food_type != "":
          food_type = new_food_type
        if new_price != "":
          price = new_price


        # Wait for delay amount of seconds before giving output
        time.sleep(delay)

        # If there is a preference left unspecified, ask user for preference
        if area == "":
          if formal:
            print("What part of town do you have in mind?")
          else:
            print("What area of town?")
        elif food_type == "":
          if formal:
            print("What kind of food would you like?")
          else:
            print("What type of food?")
        elif price == "":
          if formal:
            print("Would you like something in the cheap , moderate , or expensive price range?")
          else:
            print("what price range do you want?")
        # Go to recommendation state once all preferences have been given
        else:
            state = 3

  # State 3 is a one pass state that will send to user back to state 2 or 4 depending if on the recommendation
  if state == 3:

    # Result should be stored into recommendation (array) and set the index of this array to 0
    recommendations = get_recommendations(area, food_type, price,recommendations)
    index = 0

    #### This is where to implement 1C
    while state == 3:

    # Ask user for additional requirements
      if formal:
        print("Do you have any additional requirements?") # Options are food quality, crowdedness, length of stay
      else:
        print("anything else you want?")
      user_input = input()

      # Wait for delay amount of seconds before giving output
      time.sleep(delay)

      # If no additional requirements, then skip this part
      if pred_dialog(user_input.lower()) == "deny" or (user_input.lower()) == "no":
        state = 4
      else: 
        food_quality = []
        crowdedness = []
        length_stay = []
        if user_input == 'romantic':
          crowdedness = 'not busy'
          length_stay='long'
          recommendations=get_options(food_quality,crowdedness,length_stay,recommendations)
        elif user_input=='children' :
          crowddedness = 'not busy'
          recommendations=get_options(food_quality,crowdedness,length_stay,recommendations)
        else:
          food_quality, crowdedness, length_stay = additional_requirements(user_input.lower())
          #search options in recommanded restuarant 
          recommendations=get_options(food_quality,crowdedness,length_stay,recommendations)
          state = 4


  # State 4
  if state == 4:
    while state == 4:

      # Wait for delay amount of seconds before giving output
      time.sleep(delay)

      # If there is only 1 recommendation, recommend it and go to request state
      if len(recommendations) == 1 and index == 0:
        if formal:
          print("There is a restuarant called %s that is located in %s and its prices are %s" %(recommendations['restaurantname'].values[index],recommendations['area'].values[index],price))
        else:
          print("I found %s in the %s area with %s prices" %(recommendations['restaurantname'].values[index],recommendations['area'].values[index],price))
        state = 5

      # If there are multiple recommendations, recommend random one and go to request state
      elif len(recommendations) > 1:
        recommendations=recommendations.sample()
        if formal:
          print("There is a restuarant called %s that is located in %s and its prices are %s" %(recommendations['restaurantname'].values[index],recommendations['area'].values[index],price))
        else:
          print("I found %s in the %s area with %s prices" %(recommendations['restaurantname'].values[index],recommendations['area'].values[index],price))
        state = 5

      # If there are no recommendations, tell the user and allow for new preferences by going to state 2
      else:
        if formal:
          print("I'm sorry but there is no", price ,"priced restaurant in the", area, "serving", food_type, "food")
        else: 
          print("Haven't found any restaurants with your preferences. Enter new preferences")
        # Initiate preferences
        area = ""
        food_type = ""
        price = ""
        state = 2


  # State 4 handles all the request from the user after a recommendation
  while state == 5:
    user_input = input()

    # Analyse user_input request
    input_type = pred_dialog(user_input)
    print(input_type)

    # Wait for delay amount of seconds before giving output
    time.sleep(delay)

    # If the user input is a dialog_act bye statement, close the conversation
    if "bye" in user_input:
      dialog_active = False
      if formal:
        print("Thank you for using my as you recommendation system. I hope to see you back soon!")
      else:
        print("Bye!")

    # Print request result, this can update the recommendation
    elif input_type == "request":

      # Check if user asks for phone number and answer
      if "number" in user_input:
        if formal:
          print("The phone number of this restaurant is ",recommendations['phone'].values[index])
        else:
          print("phone number is %s",recommendations['phone'].values[index])

      # Check if user asks for address and answer
      if "address" in user_input:
        if formal:
          print("The address of this restaurant is ",recommendations['addr'].values[index]) 
        else:
          print("address is %s",recommendations['addr'].values[index])  

    # If the user wants another restaurant
    elif input_type == "reqalts" or input_type == "reqmore":
      index += 1
      state = 4
  
    # If the system did not understand the request tell the user their selected restaurand and that it is nice
    else:
      print(recommendations['restaurantname'].values[index]," is a great restaurant ")
