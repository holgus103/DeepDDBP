import numpy
# descending, spades, hearts, diamonds, clubs
# east, north, west, south
def estimate_points(values, d, bonuses = []):
    players = []
    # split by players and by colors
    for i in range(0, 4):
        player = values[i*52:(i+1)*52];
        p = [];
        for j in range(0, 4):
            p.append(player[j*13:(j+1)*13]);
        players.append(p);
    sum = 0;
# get points for north
    for v in players[1]:
        sum += numpy.sum(numpy.multiply(v, d));
# get points for south
    for v in players[3]:
        sum += numpy.sum(numpy.multiply(v, d));
    for c in bonuses:
        sum += c(players[1]);
        sum += c(players[3]);
    return sum;


def wpc_dict():
    return ([4, 3, 2, 1, 0] + numpy.repeat(0, 8).tolist());

def bamberger_dict():
    return ([7, 5, 3, 1, 0] + numpy.repeat(0, 8).tolist());

def collet_dict():
    return  ([4, 3, 2, 0.5, 0.5] + numpy.repeat(0, 8).tolist());

def akq_points_dict():
    return ([4, 3, 2, 0, 0] + numpy.repeat(0, 8).tolist());


def assert_system(hand):
    bonus = 0;
    sums = [];
    sums.append(numpy.sum(hand[0]));
    sums.append(numpy.sum(hand[1]));
    sums.append(numpy.sum(hand[2]));
    sums.append(numpy.sum(hand[3]));
    # add +2 for every void
    for val in sums:
        if val == 0:
            bonus += 2;
        if val == 1:
            bonus += 1;
        if val >= 5:
            bonus += 1; 
    return bonus;


   
    


    

  
