Ex-1
1) Consider that the probabilities for a person is affected by fever, cough given fever and flu given fever and cough are given. Implement Bayesian Belief network to find that the person affected by flu given he is affected from fever and cough. [P(Flu=Yes | Fever=Yes, Cough=Yes)].

P_fever = {'Yes':0.1, 'No':0.9}
P_cough_given_fever = {
    'Yes': {'Yes':0.7, 'No':0.3},
    'No':{'Yes':0.2,'No':0.8}
}
P_flu_given_fever_cough = {
    'Yes':{
        'Yes': {'Yes':0.8, 'No':0.2},
        'No':{'Yes':0.6,'No':0.4}
    },
    'No':{
        'Yes': {'Yes':0.1, 'No':0.9},
        'No':{'Yes':0.05,'No':0.95}
    }
}

#Bayes Theorem P(A|B) = (P(B|A)*P(A)) / P(B)
#P(A|B, C) = (P(B|A) * P(C|A) * P(A)) / P(B,C)
fever = cough = flu = 'Yes'
P_BC = P_fever[fever] * P_cough_given_fever[fever][cough]
P_B_given_A = P_flu_given_fever_cough[fever][cough][flu]
P_C_given_A = P_cough_given_fever[fever][cough]
P_A = P_fever[fever]
P_A_given_BC = (P_B_given_A * P_C_given_A * P_A)/P_BC
print(f"P((Flu=Yes)|Fever=Yes, Cough=Yes) = {P_A_given_BC:.2f}")




















2) Given probability that it is raining [R], probability that the sprinkler is on [S] and the probability that the grass is wet [W] given it is raining and sprinkler is on. Implement a Bayesian Belief Network to compute the probability that it is raining given that the grass is wet, i.e. P(R=Yes∣W=Yes). 

# Probability of Rain (P(R))
P_rain = {
    'Yes': 0.3,  # Probability that it is raining
    'No': 0.7    # Probability that it is not raining
}
# Probability of Sprinkler (P(S))
P_sprinkler = {
    'Yes': 0.5,  # Probability that the sprinkler is on
    'No': 0.5    # Probability that the sprinkler is off
}
# Probability of Wet Grass given Rain and Sprinkler (P(W | R, S))
P_wet_grass_given_rain_sprinkler = {
    ('Yes', 'Yes'): 0.99,  # P(W | R=Yes, S=Yes)
    ('Yes', 'No'): 0.90,   # P(W | R=Yes, S=No)
    ('No', 'Yes'): 0.80,   # P(W | R=No, S=Yes)
    ('No', 'No'): 0.00     # P(W | R=No, S=No)
}


P_W = (
    P_wet_grass_given_rain_sprinkler[('Yes', 'Yes')] * P_rain['Yes'] * P_sprinkler['Yes'] +
    P_wet_grass_given_rain_sprinkler[('Yes', 'No')] * P_rain['Yes'] * P_sprinkler['No'] +
    P_wet_grass_given_rain_sprinkler[('No', 'Yes')] * P_rain['No'] * P_sprinkler['Yes'] +
    P_wet_grass_given_rain_sprinkler[('No', 'No')] * P_rain['No'] * P_sprinkler['No']
)

P_W_given_R_yes = (
    P_wet_grass_given_rain_sprinkler[('Yes', 'Yes')] * P_sprinkler['Yes'] +
    P_wet_grass_given_rain_sprinkler[('Yes', 'No')] * P_sprinkler['No']
)

numerator = P_W_given_R_yes * P_rain['Yes']
P_R_given_W = numerator / P_W
print(f"Probability that it is raining given that the grass is wet: {P_R_given_W:.4f}")













3) Implement a Bayesian Belief Network to classify emails as spam [S] based on whether they contain an offer [O] and the word "free" [F], given the prior and conditional probabilities. Calculate P(S=Yes∣O=Yes, F=Yes).

# Prior probabilities of Spam (P(S))
P_spam = {
    'Yes': 0.4,  # Probability that the email is spam
    'No': 0.6    # Probability that the email is not spam
}

# Conditional probabilities
P_offer_given = {
    'Spam': 0.9,      # P(O | S)
    'Not_Spam': 0.1   # P(O | ¬S)
}

P_free_given = {
    'Spam': 0.8,      # P(F | S)
    'Not_Spam': 0.2   # P(F | ¬S)
}

# Calculate the probability of the email containing both an offer and the word "free"
# P(O and F | S) = P(O | S) * P(F | S)
P_offer_and_free_given_spam = P_offer_given['Spam'] * P_free_given['Spam']

# P(O and F | ¬S) = P(O | ¬S) * P(F | ¬S)
P_offer_and_free_given_not_spam = P_offer_given['Not_Spam'] * P_free_given['Not_Spam']

# Total probability of the email containing both an offer and "free" (P(O and F))
P_offer_and_free = (
    P_offer_and_free_given_spam * P_spam['Yes'] +
    P_offer_and_free_given_not_spam * P_spam['No']
)

# Compute the posterior probability P(S | O and F)
P_spam_given_offer_and_free = (
    P_offer_and_free_given_spam * P_spam['Yes'] / P_offer_and_free
)

# Output the result
print(f"Probability that the email is spam given it contains an offer and 'free': {P_spam_given_offer_and_free:.4f}")









4) Implement Bayesian Belief Network to classify student performance based on study habits, including study hours [H] and sleep quality [Q]. Given the prior and conditional probabilities, calculate the probability that a student will pass [P] if they study for a lot of hours [High H] but have poor sleep quality, i.e. P(P=Pass |  H=High, Q=Poor)

# Prior probabilities
P_H = {
    'High': 0.5,  # Probability of high study hours (P(H = High))
    'Low': 0.5    # Probability of low study hours (P(H = Low))
}

P_Q = {
    'Good': 0.7,  # Probability of good sleep quality (P(Q = Good))
    'Poor': 0.3   # Probability of poor sleep quality (P(Q = Poor))
}

# Conditional probabilities
P_pass_given = {
    ('High', 'Good'): 0.95,  # P(Pass | High H, Good Q)
    ('High', 'Poor'): 0.70,   # P(Pass | High H, Poor Q)
    ('Low', 'Good'): 0.60,    # P(Pass | Low H, Good Q)
    ('Low', 'Poor'): 0.20     # P(Pass | Low H, Poor Q)
}

# To compute the probability of passing for a student with high study hours and poor sleep quality
study_hours = 'High'
sleep_quality = 'Poor'

# Compute the probability of passing
P_pass = P_pass_given[(study_hours, sleep_quality)] * P_H[study_hours] * P_Q[sleep_quality]

# Output the result
print(f"Probability that a student will pass given high study hours and poor sleep quality: {P_pass:.4f}")
