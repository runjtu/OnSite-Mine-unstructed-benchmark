"""

Hybrid A* path planning for onsite

author: tpr@sjtu

"""

import heapq
import numpy as np
import math
from math import sqrt, cos, sin, tan, pi

try:
    import hybrid_a_star.reeds_shepp_path_planning as rs
except:
    raise


XY_GRID_RESOLUTION = 2.0  # [m]
YAW_GRID_RESOLUTION = np.deg2rad(15.0)  # [rad]
MOTION_RESOLUTION = 0.1  # [m] path interporate resolution
N_STEER = 20  # number of steer command

WB = 5.6  #轴距
W = 5.5  # width of car
LF = 7.2  # distance from rear to vehicle front end
LB = 1.6  # distance from rear to vehicle back end
MAX_STEER = np.deg2rad(30)  # [rad] maximum steering angle

SB_COST = 100.0  # switch back penalty cost
BACK_COST = 5.0  # backward penalty cost
STEER_CHANGE_COST = 5.0  # steer angle change penalty cost
STEER_COST = 1.0  # steer angle change penalty cost
H_COST = 5.0  # Heuristic cost



class Node:

    def __init__(self, xind, yind, yawind, direction,
                 xlist, ylist, yawlist, directions,
                 steer=0.0, pind=None, cost=None):
        self.xind = xind
        self.yind = yind
        self.yawind = yawind
        self.direction = direction
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directions = directions
        self.steer = steer
        self.pind = pind
        self.cost = cost


class Path:

    def __init__(self, xlist, ylist, yawlist, directionlist, cost):
        self.xlist = xlist
        self.ylist = ylist
        self.yawlist = yawlist
        self.directionlist = directionlist
        self.cost = cost

class Config:

    def __init__(self, observation, xyreso, yawreso):
        min_x_m = observation['test_setting']['x_min']
        min_y_m = observation['test_setting']['y_min']
        max_x_m = observation['test_setting']['x_max']
        max_y_m = observation['test_setting']['y_max']

        self.minx = round(min_x_m / xyreso)
        self.miny = round(min_y_m / xyreso)
        self.maxx = round(max_x_m / xyreso)
        self.maxy = round(max_y_m / xyreso)

        self.xw = round(self.maxx - self.minx)
        self.yw = round(self.maxy - self.miny)

        self.minyaw = round(- math.pi / yawreso) - 1
        self.maxyaw = round(math.pi / yawreso)
        self.yaww = round(self.maxyaw - self.minyaw)


def calc_motion_inputs():

    for steer in np.concatenate((np.linspace(-MAX_STEER, MAX_STEER, N_STEER),[0.0])):
        for d in [1]:
            yield [steer, d]

def check_car_collision(xlist, ylist, yawlist, collision_lookup,observation):
    local_x_range = observation['hdmaps_info']['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_x_range']
    local_y_range = observation['hdmaps_info']['image_mask'].bitmap_info['bitmap_mask_PNG']['UTM_info']['local_y_range']
    for x, y, yaw in zip(xlist, ylist, yawlist):
        if collision_lookup.collision_detection(x-local_x_range[0],
                                                y-local_y_range[0],
                                                yaw,
                                                observation['hdmaps_info']['image_mask'].image_ndarray):
            return True
    return False

def pi_2_pi(angle):
    return (angle + pi) % (2 * pi) - pi

def move(x, y, yaw, distance, steer, L=WB):
    x += distance * cos(yaw)
    y += distance * sin(yaw)
    yaw += pi_2_pi(distance * tan(steer) / L)  # distance/2

    return x, y, yaw

def get_neighbors(current, config,collision_lookup,observation):

    for steer, d in calc_motion_inputs():
        node = calc_next_node(current, steer, d, config, collision_lookup,observation)
        if node and verify_index(node, config):
            yield node


def calc_next_node(current, steer, direction, config, collision_lookup,observation):

    x, y, yaw = current.xlist[-1], current.ylist[-1], current.yawlist[-1]

    arc_l = XY_GRID_RESOLUTION * 1.5
    xlist, ylist, yawlist = [], [], []
    for dist in np.arange(0, arc_l, MOTION_RESOLUTION):
        x, y, yaw = move(x, y, yaw, MOTION_RESOLUTION * direction, steer)
        xlist.append(x)
        ylist.append(y)
        yawlist.append(yaw)
    if check_car_collision(xlist, ylist, yawlist, collision_lookup,observation):
        return None

    d = direction
    xind = round(x / XY_GRID_RESOLUTION)
    yind = round(y / XY_GRID_RESOLUTION)
    yawind = round(yaw / YAW_GRID_RESOLUTION)

    addedcost = 0.0

    if d != current.direction:
        addedcost += SB_COST

    # steer penalty
    addedcost += STEER_COST * abs(steer)

    # steer change penalty
    addedcost += STEER_CHANGE_COST * abs(current.steer - steer)

    cost = current.cost + addedcost + arc_l

    node = Node(xind, yind, yawind, d, xlist,
                ylist, yawlist, [d],
                pind=calc_index(current, config),
                cost=cost, steer=steer)

    return node


def is_same_grid(n1, n2):
    if n1.xind == n2.xind and n1.yind == n2.yind and n1.yawind == n2.yawind:
        return True
    return False


def analytic_expantion(current, goal, c, collision_lookup, observation,bo_alp_in):

    sx = current.xlist[-1]
    sy = current.ylist[-1]
    syaw = current.yawlist[-1]

    gx = goal.xlist[-1]
    gy = goal.ylist[-1]
    gyaw = goal.yawlist[-1]

    max_curvature = math.tan(MAX_STEER) / WB
    paths = rs.calc_paths(sx, sy, syaw, gx, gy, gyaw,
                          max_curvature, step_size=MOTION_RESOLUTION)

    if not paths:
        return None

    best_path, best = None, None
    for path in paths:
        if not check_car_collision(path.x, path.y, path.yaw, collision_lookup,observation):
            l_back = 0
            b_num = 0
            for l in path.lengths:
                if l < 0:  # backward
                    l_back += abs(l)
                    b_num += 1
            # print(l_back,b_num)
            if bo_alp_in:
                if path.lengths[-1]<0 and path.lengths[-2]>0 and l_back<150 and b_num <2:
                    cost = calc_rs_path_cost(path)
                    if not best or best > cost:
                        best = cost
                        best_path = path
            else:
                if b_num == 0 and l_back == 0:
                    cost = calc_rs_path_cost(path)
                    if not best or best > cost:
                        best = cost
                        best_path = path
    return best_path


def update_node_with_analystic_expantion(current, goal,
                                         c, collision_lookup, observation,bo_alp_in):
    apath = analytic_expantion(current, goal, c, collision_lookup, observation,bo_alp_in)
    # print("use rs")
    if apath:
        fx = apath.x[1:]
        fy = apath.y[1:]
        fyaw = apath.yaw[1:]

        fcost = current.cost + calc_rs_path_cost(apath)
        fpind = calc_index(current, c)

        fd = []
        for d in apath.directions[1:]:
            fd.append(d >= 0)

        fsteer = 0.0
        fpath = Node(current.xind, current.yind, current.yawind,
                     current.direction, fx, fy, fyaw, fd,
                     cost=fcost, pind=fpind, steer=fsteer)
        return True, fpath

    return False, None


def calc_rs_path_cost(rspath):

    cost = 0.0
    for l in rspath.lengths:
        if l >= 0:  # forward
            cost += l
        else:  # back
            cost += abs(l) * BACK_COST
    # swich back penalty
    for i in range(len(rspath.lengths) - 1):
        if rspath.lengths[i] * rspath.lengths[i + 1] < 0.0:  # switch back
            cost += SB_COST

    # steer penalyty
    for ctype in rspath.ctypes:
        if ctype != "S":  # curve
            cost += STEER_COST * abs(MAX_STEER)

    # ==steer change penalty
    # calc steer profile
    nctypes = len(rspath.ctypes)
    ulist = [0.0] * nctypes
    for i in range(nctypes):
        if rspath.ctypes[i] == "R":
            ulist[i] = - MAX_STEER
        elif rspath.ctypes[i] == "L":
            ulist[i] = MAX_STEER

    for i in range(len(rspath.ctypes) - 1):
        cost += STEER_CHANGE_COST * abs(ulist[i + 1] - ulist[i])

    return cost


def hybrid_a_star_planning(start, goal, collision_lookup, observation, xyreso, yawreso,bo_alp_in):
    """
    start
    goal
    xyreso: grid resolution [m]
    yawreso: yaw angle resolution [rad]
    """
    print("Start Hybrid A* planning!")
    start[2], goal[2] = rs.pi_2_pi(start[2]), rs.pi_2_pi(goal[2])
    # tox, toy = ox[:], oy[:]

    # obkdtree = KDTree(np.vstack((tox, toy)).T)

    config = Config(observation, xyreso, yawreso)

    nstart = Node(round(start[0] / xyreso), round(start[1] / xyreso), round(start[2] / yawreso),
                  True, [start[0]], [start[1]], [start[2]], [True], cost=0)
    ngoal = Node(round(goal[0] / xyreso), round(goal[1] / xyreso), round(goal[2] / yawreso),
                 True, [goal[0]], [goal[1]], [goal[2]], [True])

    openList, closedList = {}, {}

    pq = []
    openList[calc_index(nstart, config)] = nstart
    heapq.heappush(pq, (calc_cost(nstart,goal),calc_index(nstart, config)))
    iter_num = 0
    while True:
        iter_num +=1
        if not openList:
            print("Cannot find path, No open set!")
            return None

        cost, c_id = heapq.heappop(pq)
        # print(cost)
        if c_id in openList:
            current = openList.pop(c_id)
            closedList[c_id] = current
        else:
            continue
        # dist = sqrt((current.xlist[-1] - goal[0]) **2 + (current.ylist[-1] - goal[1]) **2)
        # if dist<50:
        # print("use rs start")
        isupdated, fpath = update_node_with_analystic_expantion(
            current, ngoal, config, collision_lookup, observation,bo_alp_in)

        if isupdated:
            print("Use rs successfully!")
            break

        for neighbor in get_neighbors(current, config, collision_lookup,observation):
            neighbor_index = calc_index(neighbor, config)
            if neighbor_index in closedList:
                continue
            if neighbor not in openList \
                    or openList[neighbor_index].cost > neighbor.cost:
                heapq.heappush(pq, (calc_cost(neighbor,goal),neighbor_index))
                openList[neighbor_index] = neighbor
        if iter_num>1000:
            print("Cannot find path, beyond limit!")
            return None
    path = get_final_path(closedList, fpath, nstart, config)
    return path


def calc_cost(n,goal):
    cost = sqrt((n.xlist[0] - goal[0]) **2 + (n.ylist[0] - goal[1]) **2)
    return n.cost + H_COST * cost


def get_final_path(closed, ngoal, nstart, config):
    rx, ry, ryaw = list(reversed(ngoal.xlist)), list(
        reversed(ngoal.ylist)), list(reversed(ngoal.yawlist))
    direction = list(reversed(ngoal.directions))
    # print(len(rx),len(direction))
    nid = ngoal.pind
    finalcost = ngoal.cost

    while nid:
        n = closed[nid]
        # print(n.directions)
        rx.extend(list(reversed(n.xlist)))
        ry.extend(list(reversed(n.ylist)))
        ryaw.extend(list(reversed(n.yawlist)))
        if n.directions == True or 1:
            direction.extend(list(reversed([True]*len(n.xlist))))
        else:
            direction.extend(list(reversed([False]*len(n.xlist))))
        # direction.extend([True])
        nid = n.pind
    # print(len(rx),len(direction))
    rx = list(reversed(rx))
    ry = list(reversed(ry))
    ryaw = list(reversed(ryaw))
    direction = list(reversed(direction))
    # print(direction)
    # adjust first direction
    direction[0] = direction[1]

    path = Path(rx, ry, ryaw, direction, finalcost)

    return path


def verify_index(node, c):
    xind, yind = node.xind, node.yind
    if xind >= c.minx and xind <= c.maxx and yind >= c.miny \
            and yind <= c.maxy:
        return True

    return False


def calc_index(node, c):
    ind = (node.yind - c.miny) * c.xw + (node.xind - c.minx)

    if ind <= 0:
        print("Error(calc_index):", ind)

    return ind
