#!/usr/bin/env python
import string
from collections import defaultdict
from Queue import PriorityQueue
from math import radians, cos, sin, asin, sqrt
import sys

# All the heuristic logics, and ways of handling the data inconsistency explained below.
# Assuming the end_city given as input will have lat, lon values as mentioned in piazza.. otherwise this might fail


'''
    Logic used for A-Star:
        1) Heuristic for Distance:
            -- Calculated heuristic as - Since the displacement is always <= to the road distance, calculated displacement between the
               current city to goal city using the latitudes, longitudes given in the city-gps.txt file and used it as heuristic value.
            -- Used haversine formula to calculate the displacement -- Mentioned the reference in the 'calDisplacement' function.

            Handling the data inconsistency:
                -- Problem with calculating the displacements was that the given data doesn't have lat, lon values of all cities.
                -- In that case, I considered the city which is nearest to it by road distance by iterating through it's succesors/predecessors.
                -- Again, one more problem here. If none of it's nearby cities have the lat, lon data; then I considered the lat, lon average
                   of that state in which city is present and calculated distance using that.
                -- Yet again, faced few problems because of invalid state names like Jct_34&56. There are cases where the none of the cities in a
                   particular state are mentioned in the city-gps.txt file. In that case, average can't be calculated.
                -- So, to tackle the above problem, while iterating, i'm storing the previous city in the route and using it's lat, lon values.

        2) Heuristic for Time:
            -- Calculated heuristic as - Distance(heuristic value as caluculated above) / Average speed of all routes in the given data.
            -- Average speeds of all routes given in the data seemed to be the best option as the denominator, as we are neither aware of the upcoming route,
               nor can rely on the route followed till now, as the speeds keeps changing based on states and highways.

            Handling the data inconsistency:
                -- Problem here was that few routes were missing speed values or have the speed values as '0'.
                -- So to handle this problem, i've replaced 0s and missing speeds with average speed in that particular highway,
                   rather than overall avergae. If that highway does not occur in any other route, then using the over all avergae speed instead.

        3) Heuristic for Segments:
            -- Calculated the heuristic as - Distance(heuristic value as caluculated above) / Average distance of all routes
            -- Intuition behind this is that, by diving heuristic distance value with average distance, we get
               the estimated number of segments.

    -- Although the heuristics seems to be good enough and data inconsisties handled properly, but still was not able to get optimal answers
       as compared to UCS.
    -- After thorough investigation, I have identified that the latitudes, longitudes data given in the file itself are wrong or the road
       distance values in the road-segments.txt file are wrong. They give the displacement values higher than the road distances, thus making
       the heuristic inadmissible and not able to produce optimal results.

    '''

def solve_ids(start_city, end_city, algo, cost):
    for i in range(10000):
        fringe = []
        distance_so_far = 0
        time_so_far = 0
        route_so_far = start_city
        fringe.append((start_city, route_so_far, 0, distance_so_far, time_so_far))
        visited = defaultdict(list)
        while len(fringe):
            if fringe[len(fringe) - 1][2] <= i:
                (state, route_so_far, depth_so_far, distance_so_far, time_so_far) = fringe.pop()
                if state == end_city:
                    if cost == "segments":
                        return "yes " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
                    else:
                        return "no " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
                if not visited[state]:
                    visited[state] = True
                    if depth_so_far + 1 <= i:
                        for city in succOfCity[state]:
                            fringe.append((city[0], route_so_far + " " + city[0], depth_so_far + 1, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]) ))
    return False

def solve_dfs(start_city, end_city, algo, cost):
    fringe = []
    route_so_far = start_city
    distance_so_far = 0
    time_so_far = 0
    fringe.append((start_city, route_so_far, distance_so_far, time_so_far))
    visited = defaultdict(list)
    while len(fringe):
        (state, route_so_far, distance_so_far, time_so_far) = fringe.pop()
        if state == end_city:
            return "no " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
        if not visited[state]:
            visited[state] = True
            for city in succOfCity[state]:
                fringe.append((city[0], route_so_far + " " + city[0], distance_so_far + city[1], time_so_far + float(city[1])/float(city[2])))
    return False

def solve_bfs(start_city, end_city, algo, cost):
    fringe = []
    route_so_far = start_city
    distance_so_far = 0
    time_so_far = 0
    fringe.append((start_city, route_so_far, distance_so_far, time_so_far))
    visited = defaultdict(list)
    while len(fringe):
        (state, route_so_far, distance_so_far, time_so_far) = fringe.pop(0)
        if state == end_city:
            if cost == "segments":
                return "yes " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
            else:
                return "no " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
        if not visited[state]:
            visited[state] = True
            for city in succOfCity[state]:
                fringe.append((city[0], route_so_far + " " + city[0], distance_so_far + city[1], time_so_far + float(city[1])/float(city[2])))
    return False

def solve_ucs(start_city, end_city, algo, cost):
    fringe = PriorityQueue()
    route_so_far = start_city
    cost_so_far = 0
    distance_so_far = 0
    time_so_far = 0
    fringe.put((0, (start_city, route_so_far, cost_so_far, distance_so_far, time_so_far)))
    visited = defaultdict(list)
    visited[start_city] = True
    while fringe.qsize() > 0:
        (state, route_so_far, cost_so_far, distance_so_far, time_so_far) = fringe.get()[1]
        visited[state] = True
        if state == end_city:
            return "yes " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
        else:
            for city in succOfCity[state]:
                if not visited[city[0]]:
                    if cost == "distance":
                        current_cost = city[1]
                        fringe.put((current_cost + cost_so_far, (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
                    elif cost == "time":
                        current_cost = float(city[1]) / float(city[2])
                        fringe.put((current_cost + cost_so_far, (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
                    elif cost == "segments":
                        current_cost = 1
                        fringe.put((current_cost + cost_so_far, (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
    return False



def calcDisplacement(fromCity, toCity):
    """
        Haversine method to Calculate the great circle distance between two points
        on the earth (specified in decimal degrees)
    """
    # Referred from the below link.
    # https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
    global prev_state
    if latlon[fromCity]:
        lon1 = latlon[fromCity][0]
        lat1 = latlon[fromCity][1]
    else:
        #If lat, lon fof cities missing

        if fromCity == "" or len(citiesInState[fromCity.split(",")[1]]) == 0:
            # If nearest city with lon, lat found in it's succesors and unable to calc state average because of invalid name or has no cities in the state with lat, lon data
            lon1 = LonsofState[prev_state] / len(citiesInState[prev_state])
            lat1 = LatsofState[prev_state] / len(citiesInState[prev_state])
        else:
            # If nearest city with lon, lat found in it's succesors, calclate state avg
            lon1 = LonsofState[fromCity.split(",")[1]] / len(citiesInState[fromCity.split(",")[1]])
            lat1 = LatsofState[fromCity.split(",")[1]] / len(citiesInState[fromCity.split(",")[1]])
    if latlon[toCity]:
        lon2 = latlon[toCity][0]
        lat2 = latlon[toCity][1]
    else:
        lon2 = LonsofState[toCity.split(",")[1]] / len(citiesInState[toCity.split(",")[1]])
        lat2 = LatsofState[toCity.split(",")[1]] / len(citiesInState[toCity.split(",")[1]])

    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    #global prev_state
    if fromCity != "":
        prev_state = fromCity.split(",")[1]
    # haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    r = 3956  # Radius of earth in kilometers. Use 3956 for miles
    return c * r

def findClosestCity(city):
    min = 10000
    nearestCity = ""
    roadDist = 0
    for nearbyCities in succOfCity[city]:
        if latlon[nearbyCities[0]]:
            roadDist = nearbyCities[1]
            if roadDist < min:
                min = roadDist
                nearestCity = nearbyCities[0]
        else:
            continue
    return nearestCity
    

def heuristic_astar(city):

    if latlon[city] and latlon[end_city]:
        return calcDisplacement(city, end_city)
    elif latlon[city] and not latlon[end_city]:
        return calcDisplacement(city, findClosestCity(end_city))
    elif not latlon[city] and latlon[end_city]:
        return calcDisplacement(findClosestCity(city), end_city)
    else:
        return calcDisplacement(findClosestCity(city), findClosestCity(end_city))


def solve_astar(start_city, end_city, algo, cost):
    
    fringe = PriorityQueue()
    route_so_far = start_city
    cost_so_far = 0
    distance_so_far = 0
    time_so_far = 0
    fringe.put((0, (start_city, route_so_far, cost_so_far, distance_so_far, time_so_far)))
    visited = defaultdict(list)
    visited[start_city] = True
    while fringe.qsize() > 0:
        (state, route_so_far, cost_so_far, distance_so_far, time_so_far) = fringe.get()[1]
        visited[state] = True
        if state == end_city:
            return "yes " + str(distance_so_far) + " " + str(time_so_far) + " " + route_so_far
        else:
            for city in succOfCity[state]:
                if not visited[city[0]]:
                    if cost == "distance":
                        current_cost = city[1]
                        fringe.put((current_cost + cost_so_far + heuristic_astar(city[0]), (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
                    elif cost == "time":
                        current_cost = float(city[1]) / float(city[2])
                        fringe.put((current_cost + cost_so_far + float(heuristic_astar(city[0]))/float(avg_speed), (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
                    elif cost == "segments":
                        current_cost = 1
                        fringe.put((current_cost + cost_so_far + heuristic_astar(city[0])/float(avg_dist), (city[0], route_so_far + " " + city[0], current_cost + cost_so_far, distance_so_far + city[1], time_so_far + float(city[1])/float(city[2]))))
    return False

succOfCity = defaultdict(list)
speedsofHighway = defaultdict(list)
sumOfSpeeds = 0
numOfRoutes = 0
sumOfDist = 0

with open('road-segments.txt', 'r') as file:
    for line in file:
        if not len(line.split()) == 4 and int(line.split()[3]) != 0:
            succOfCity[line.split()[0]].append((line.split()[1], int(line.split()[2]), int(line.split()[3]), line.split()[4]))
            succOfCity[line.split()[1]].append((line.split()[0], int(line.split()[2]), int(line.split()[3]), line.split()[4]))
            speedsofHighway[line.split()[4]].append(int(line.split()[3]))
        numOfRoutes = numOfRoutes + 1

with open('road-segments.txt', 'r') as file:
    for line in file:
        sumOfDist = sumOfDist + int(line.split()[2])
        if len(line.split()) == 4 or int(line.split()[3]) == 0:
            if not speedsofHighway[line.split()[3]]:
                speedsofHighway[line.split()[3]].append(int(30))
                sumOfSpeeds = sumOfSpeeds + 30
            else:
                sumOfSpeeds = sumOfSpeeds + int(sum(speedsofHighway[line.split()[3]])/len(speedsofHighway[line.split()[3]]))
            succOfCity[line.split()[0]].append((line.split()[1], int(line.split()[2]), int(sum(speedsofHighway[line.split()[3]])/len(speedsofHighway[line.split()[3]])), line.split()[3]))
            succOfCity[line.split()[1]].append((line.split()[0], int(line.split()[2]), int(sum(speedsofHighway[line.split()[3]])/len(speedsofHighway[line.split()[3]])), line.split()[3]))
        else:
            sumOfSpeeds = sumOfSpeeds + int(line.split()[3])

avg_speed = float(sumOfSpeeds)/float(numOfRoutes)
avg_dist = float(sumOfDist)/float(numOfRoutes)
latlon = defaultdict(list)
citiesInState = defaultdict(list)
LonsofState = defaultdict(float)
LatsofState = defaultdict(float)

with open('city-gps.txt', 'r') as file:
    for line in file:
        citiesInState[(line.split()[0]).split(",")[1]].append(line.split()[0])
        if not LonsofState[(line.split()[0]).split(",")[1]]:
            LonsofState[(line.split()[0]).split(",")[1]] = 0
        if not LatsofState[(line.split()[0]).split(",")[1]]:
            LatsofState[(line.split()[0]).split(",")[1]] = 0
        LonsofState[(line.split()[0]).split(",")[1]] = LonsofState[(line.split()[0]).split(",")[1]] + float(line.split()[1])
        LatsofState[(line.split()[0]).split(",")[1]] = LatsofState[(line.split()[0]).split(",")[1]] + float(line.split()[2])
        latlon[line.split()[0]].append((float(line.split()[1])))
        latlon[line.split()[0]].append((float(line.split()[2])))

start_city = sys.argv[1]
end_city = sys.argv[2]
algo = sys.argv[3]
cost = sys.argv[4]
prev_state = start_city.split(",")[1]
if algo == "bfs":
    print(solve_bfs(start_city, end_city, algo, cost))
if algo == "uniform":
    print(solve_ucs(start_city, end_city, algo, cost))
if algo == "dfs":
    print(solve_dfs(start_city, end_city, algo, cost))
if algo == "ids":
    print(solve_ids(start_city, end_city, algo, cost))
if algo == "astar":
    print(solve_astar(start_city, end_city, algo, cost))
