#States
# 1 preference request
# 2 ask for missing preferences
# 3 ask for additional requirements and adjust restaurant options
# 4 recommend restaurant based on preferences
# 5 listen for information requests

try:
  import '1C_Helper_Functions.py'

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
