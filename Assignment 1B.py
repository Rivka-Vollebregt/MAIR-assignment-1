"""# Part 1B
## Keyword matching
Receives user input string and finds keywords
"""

# Determines levenshteinDistance between two elements
import numpy as np

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
    return return_str.join(closestWords)
  else:
    return None

# Function to find keyword user preferences for area, food and price in string
# Utterance has to be a list of string words
def keyword(utterance):

  # Find area
  area = []

  # Possible areas
  dictionary_area = ["north","south","east","west","centre"]

  for word in utterance:
    # If input word matches one in dictionary, then succesful and break
    if word in dictionary_area:
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
  # Check if there's a word close to the input word
  if len(price) < 1:
      price = closest_area(utterance,dictionary_price)

  # Return found user preferences
  return area, food_type, price

# Preference function testing
# Prepare user input to find preferences
inpt = "Hi I am looking for the Nerth restaurant with food"

# Convert input to lowercase
inpt_lower = inpt.lower()

# Find keywords
area, food_type, price = keyword(inpt_lower.split())

# Recommnedation lookup function
import pandas as pd
import numpy as np
import random

# Find a matching recommendation from database restaurant_info.csv
def get_recommendations(area, food_type, price):
  df = pd.read_csv("restaurant_info.csv") 
  recommendations=df[(df.area == area) & (df.food==food_type) & (df.pricerange==price)]
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

#States
# 1 preference request
# 2 ask for missing preferences
# 3 recommend restaurant based on preferences
# 4 listen for information requests
# 5 dialog close
# 6 cannot understand request
# 7 answer request

# Dialog system
state = 1
dialog_active = True


# Initiate preferences
area = None
food_type = None
price = None

recommendations = []

# Until user ends conversation, check for input and handle accordingly
while dialog_active:
  if state == 1:

  # Start the conversation with the welcome sentence
    print("Hello , welcome to the Cambridge restaurant system? You can ask for restaurants by area , price range or food type . How may I help you?")

  # Go to next state after asking the question
    state = 2

  # State 2 is the state to get the missing preferences and allows the user to update previous preferences
  while state == 2:
    
    # Get user input for missing preferences
    user_input = input()
    user_input = user_input.lower()
    new_area, new_food_type, new_price = keyword(user_input.split())

    # Update old preferences when new preference != None
    if new_area != None:
      area = new_area
    if new_food_type != None:
      food_type = new_food_type
    if new_price != None:
      price = new_price

    # If there is a preference left unspecified, ask user for preference
    if area == None:
      print("What part of town do you have in mind?")
    elif food_type == None:
      print("What kind of food would you like?")
    elif price == None:
      print("Would you like something in the cheap , moderate , or expensive price range?")
    # Go to recommendation state once all preferences have been given
    else:
        state = 3

  # State 3 is a one pass state that will send to user back to state 2 or 4 depending if on the recommendation
  if state == 3:

    # Result should be stored into recommendation (array)
    recommendations = get_recommendations(area, food_type, price)

    # If there is only 1 recommendation, recommend it and go to request state
    if len(recommendations) == 1:
      print("There is a restuarant called Name is %s that is located in %s and its prices are %s" %(recommendations['restaurantname'].values[0],recommendations['area'].values[0],price))
      state = 4

    # If there are multiple recommendations, recommend random one and go to request state
    elif len(recommendations) > 1:
      recommendations=recommendations.sample()
      print("There is a restuarant called Name is %s that is located in %s and its prices are %s" %(recommendations['restaurantname'].values[0],recommendations['area'].values[0],price))
      state = 4

    # If there are no recommendations, tell the user and allow for new preferences by going to state 2
    else:
      print("I'm sorry but there is no", price ,"priced restaurant in the", area, "serving", food_type, "food")
      state = 2

  # State 4 handles all the request from the user after a recommendation
  while state == 4:
    user_input = input()

    # Analyse user_input request
    input_type = dialog_type(user_input)

    # If the user input is a dialog_act bye statement, close the conversation
    if "bye" in user_input:
      state = 5
      dialog_active = False

    # Print request result, this can update the recommendation
    elif input_type == "request":

      # Check if user asks for phone number and answer
      if "number" in user_input:
        print("The phone number of this restaurant is ",recommendations['phone'].values[0])

      # Check if user asks for address and answer
      if "address" in user_input:
        print("The address of this restaurant is ",recommendations['addr'].values[0])    
  
    # If the system did not understand the request tell the user their selected restaurand and that it is nice
    else:
      print(recommendations['restaurantname'].values[0]," is a great restaurant ")
