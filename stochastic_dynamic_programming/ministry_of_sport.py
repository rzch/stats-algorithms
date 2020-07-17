# The Riddler Classic 2018-11-09: Ministry of Sport
# https://fivethirtyeight.com/features/what-are-the-odds-youd-already-have-my-number/
# Computation and Monte-Carlo simulation of optimal policy (with respect to mean) 
# based on dynamic programming

import numpy as np
import random

 #%% Function definitions-----------------------------------------------------

def generate_random_schedule(T):
    #generates a (T*(T+1)/2)x2 list of lists for a random schedule of matches between teams
    
    #teams are integers between 1 to T inclusive
    teams = list(range(1, T + 1))
    
    #generate all pairs of matches
    schedule = []
    for i in range(T):
        for j in range(i + 1, T):
            schedule.append([teams[i], teams[j]])
            
    #randomly shuffle the order
    random.shuffle(schedule)     
    
    return schedule

def generate_random_result(t1, t2):
    #generates a random result between teams based on their strength
    #returns either a 1, 2 or 0 based on whether team 1 wins, loses or draws respectively
    
    #probabilities
    t = t1 + t2 + 1
    p1 = t1/t
    p2 = t2/t
    p0 = 1/t
    
    #sample result
    r = random.random()
    if (r <= p1):
        return 1
    elif (r <= p1 + p2):
        return 2
    else:
        return 0
    
def winner_guaranteed(x, schedule):
    #checks whether a winner is guaranteed given the current points and remaining schedule
    #x is a list of current points
    #schedule is the remaining schedule
    
    T = len(x) #number of teams
    #leading team(s)
    s = np.max(x)
    imax = np.argwhere(x == s).flatten().tolist()
    
    #if there is only 1 leading team, check that no other team can catch up even if they won all their matches
    if len(imax) == 1:

        for i in range(T):
            t = i + 1
            if np.isin(i, imax):
                continue

            #add up potential maximum score for team t
            y = x[i]
            for j in range(len(schedule)):
                if (schedule[j][0] == t or schedule[j][1] == t):
                    y += 2
            
            if (y >= s):
                return False

    #if there is more than 1 leading team,
    elif len(imax) > 1:
        #check that none of the leading team has any remaining matches (thus can score no more points)
        for i in imax:
            t = i + 1
            for j in range(len(schedule)):
                if (schedule[j][0] == t or schedule[j][1] == t):
                    return False
        
        #check that no other team can catch up even if they won all their matches
        for i in range(T):
            t = i + 1
            if np.isin(i, imax):
                continue

            y = x[i]

            for j in range(len(schedule)):
                if (schedule[j][0] == t or schedule[j][1] == t):
                    y += 2

            if (y >= s):
                return False        
    
    return True

def generate_oop(T, schedule):
    #given a schedule for T teams, this function returns the index of order of play so that each team's first game is
    #at the beginning
    
    #create indices
    oop = list(range(len(schedule)))
    
    #determine index of first game for each team
    first_game = [0]*T
    for i in range(T):
        t = i + 1
        #find the index of first game for team t
        for j in range(len(schedule)):
            if (schedule[j][0] == t or schedule[j][1] == t):
                first_game[i] = j
                break
    
    #remove duplicate indices
    oop2 = list(set(first_game))
    
    #remove first game indices from original list
    for i in range(len(oop2)):
        oop.remove(oop2[i])
        
    #append remaining original indices to the indices of the first games
    [oop2.append(j) for j in oop]
    
    return oop2

def first_games(T, schedule):
    #given a schedule for T teams, this function returns the index of the first games for each team
    #(which has at least one match left to play)
    
    #determine index of first game for each team
    first_game = [-1]*T
    for i in range(T):
        t = i + 1
        #find the index of first game for team t
        for j in range(len(schedule)):
            if (schedule[j][0] == t or schedule[j][1] == t):
                first_game[i] = j
                break
    
    #remove sentinels
    first_game = [f for f in first_game if f != -1]
    #remove duplicate indices
    first_game = list(set(first_game))
    
    return first_game

def simulate_schedule(x, schedule):
    #simulates a season based on starting number of points x and remaining schedule
    #returns history of x and remaining points for each team
    
    T = len(x)
    xcurr = list.copy(x)
    xhist = []
    dhist = []
    
    #generate order of play for a random schedule
    oop = generate_oop(T, generate_random_schedule(T))
    
    #rearrange sechedule according to order of play
    schedule = [schedule[k] for k in oop]
    
    #simulate each match until a season winner/tie is guaranteed
    for j in range(len(schedule)):
        
        #simulate a match result
        result = generate_random_result(schedule[j][0], schedule[j][1])
        if (result == 1):
            xcurr[schedule[j][0] - 1] += 2
        elif (result == 2):
            xcurr[schedule[j][1] - 1] += 2
        elif (result == 0):
            xcurr[schedule[j][0] - 1] += 1
            xcurr[schedule[j][1] - 1] += 1
            
        xhist.append(list.copy(xcurr))
        
        #create the remaining schedule
        remaining_schedule = [schedule[k] for k in range(j + 1, len(schedule))]
        
        d = expected_remaining_points(T, remaining_schedule)
        dhist.append(d)
        
        #check if a winner/tie is guaranteed
        if winner_guaranteed(xcurr, remaining_schedule):
            break
    
    return xhist, dhist

def expected_remaining_points(T, schedule):
    #given remaining schedule for T teams, returns a list of the expected points for each team
    
    d = [0]*T
    
    #go through each match and update expectations for participating teams
    for j in range(len(schedule)):
        
        t1 = schedule[j][0]
        t2 = schedule[j][1]
        t = t1 + t2 + 1
        p1 = t1/t
        p2 = t2/t
        p0 = 1/t
    
        d[t1 - 1] += (2*p1 + p0)
        d[t2 - 1] += (2*p2 + p0)
        
    return d

def expected_matches_left(x, schedule):
    #given remaining schedule for T teams, returns the expected number of matches
    #until a season winner/tie is guaranteed, by following an optimal policy
    
    #trivial computation
    if (len(schedule) == 0 or winner_guaranteed(x, schedule)):
        return 0, [], [], []
    
    T = len(x) #number of teams
    
    #get indices of possible options for allowed matches
    next_day = first_games(T, schedule)

    score = [np.Inf]*len(schedule)

    #compute expected matches for each possible option
    for j in next_day:

        #compute probabilities
        t1 = schedule[j][0]
        t2 = schedule[j][1]
        t = t1 + t2 + 1
        p1 = t1/t
        p2 = t2/t
        p0 = 1/t

        #enumerate through possibilities (win/lose/draw) if this match is played
        temp_schedule = list.copy(schedule)
        temp_schedule.remove(schedule[j])   

        #team 1 wins
        xtemp1 = list.copy(x)
        xtemp1[t1 - 1] += 2                                   
        V1 = expected_matches_left(xtemp1, temp_schedule)[0]
        
        #team 2 wins
        xtemp2 = list.copy(x)
        xtemp2[t2 - 1] += 2            
        V2 = expected_matches_left(xtemp2, temp_schedule)[0]
        
        #draw
        xtemp0 = list.copy(x)
        xtemp0[t1 - 1] += 1
        xtemp0[t2 - 1] += 1           
        V0 = expected_matches_left(xtemp0, temp_schedule)[0]

        #expected matches left
        score[j] = 1 + p1*V1 + p2*V2 + p0*V0

    jmin = np.argmin(score)
    smin = score[jmin]
    
    #return the lowest expected matches and corresponding index
    return smin, jmin, score, next_day

def play_game(x, match):
    #this function returns updated scores after playing a game
    
    result = generate_random_result(match[0], match[1])
    if (result == 1): #team 1 win
        x[match[0] - 1] += 2
    elif (result == 2): #team 2 win
        x[match[1] - 1] += 2
    elif (result == 0): #draw
        x[match[0] - 1] += 1
        x[match[1] - 1] += 1
    
    return x

def simulate_season(T):
    #this function simulates a full season for T teams using the optimal policy,
    #and returns the history of scores
    
    x = [0]*T
    
    xhist = []
    
    schedule = generate_random_schedule(T)
    first_day = first_games(T, schedule)
    
    first_day_schedule = [schedule[j] for j in first_day]
    #play first day matches
    for j in first_day:
        
        match = schedule[j]
        x = play_game(x, match)
        xhist.append(list.copy(x))
        
    [schedule.remove(f) for f in first_day_schedule]
    
    #play remaining matches until winner guaranteed
    while(len(schedule) > 0):
        
        if winner_guaranteed(x, schedule):
            break
            
        smin, jmin, score, next_day = expected_matches_left(x, schedule)
        match = schedule[jmin]
        x = play_game(x, match)                
        schedule.remove(match)        
        xhist.append(list.copy(x))
                    
    return xhist

 #%% Unit test-------------------------------------------------------------

T = 4
#print('Unit Test for ' + str(T) + ' teams')
#print(expected_matches_left([0]*T, generate_random_schedule(T)))

 #%% Random selection of matches---------------------------------------------

T = 4

Nsim = 1000

Ls = [0]*Nsim
for N in range(Nsim):
    xhist, dhist = simulate_schedule([0]*T, generate_random_schedule(T))

    L = len(xhist)
    Ls[N] = L

print('Random selection')
print('Mean: ' + str(sum(Ls)/Nsim))
print('Median: ' + str(np.median(Ls)))

 #%% Optimal policy ---------------------------------------------

T = 4

Nsim = 1000

Ls = [0]*Nsim
for N in range(Nsim):
    Ls[N] = len(simulate_season(T))
    
print('Mean-optimal policy')
print('Mean: ' + str(sum(Ls)/Nsim))
print('Median: ' + str(np.median(Ls)))