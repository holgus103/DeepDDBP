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

def three_and_four(hand, is_trump = True):
    bonus = 0;
    sums = [];
    for i in range(0, 4):
        sums.append(numpy.sum(hand[i]));
    if is_trump:
        val = 4
    else:
        val = 3
    bonus += (sums[0] - val) > 0 and (sums[0] - val) or 0

    for i in range(1, 4):
            bonus += (sums[i] - 3) > 0 and (sums[i] - 3) or 0


def plus_value(hand):
    bonus = 0;
    for i in range(0, 4):
        bonus += hand[i][0] * 0.25;
        ten_with_honors = numpy.sum(hand[i][0:4]) + hnad[i][5]
        if ten_with_honors > 0:
            bonus += 0.5;
        if numpy.sum(hand[i][0:4]) >= 3 or numpy.sum(hand[i][0:3]) >= 2:
            bonus += 0.5;
    return bonus;






   
    


    

  
