 # 内置库 
import math
import statistics

# 第三方库
import numpy as np
from shapely.geometry import Point, Polygon
from typing import Dict,List,Tuple,Optional,Union
from hybrid_a_star.hybrid_a_star import hybrid_a_star_planning


# class Point2D:
#     """2D平面的位姿点."""
#     def __init__(self,x,y,yaw_rad:float=0.0):
#         self.x = x
#         self.y = y
#         self.yaw = yaw_rad


class Planner:
    """ego车的轨迹规划器.
    注:业界一般做法是 path planning + speed planning .
    """
    def __init__(self,observation):
        self._goal_x = statistics.mean(observation['test_setting']['goal']['x'])
        self._goal_y = statistics.mean(observation['test_setting']['goal']['y'])
        self._goal_yaw = observation['test_setting']['goal']['head'][0]
        self._observation = observation
    
    
    def process(self,collision_lookup,observation):
        """规划器主函数.
        注：该函数功能设计举例
        1) 进行实时轨迹规划;
        2) 路径、速度解耦方案:给出局部轨迹规划器的实时求解结果--待行驶路径、待行驶速度；
        
        输入:observation——环境信息;
        输出: 路径、速度解耦方案:给出局部轨迹规划器的实时求解--
            待行驶路径（离散点序列）、待行驶速度（与路径点序列对应的速度离散点序列）
        """
        # 设置目标速度
        target_speed = 8.0
        target_speed_backward = 10.0/3.6

        goal = [np.mean(observation['test_setting']['goal']['x']), 
                np.mean(observation['test_setting']['goal']['y']), 
                observation['test_setting']['goal']['head'][0]]
        spd_planned = []
        # path_planned = self.get_ego_reference_path_to_goal(observation)
        keypoint_ind = 0
        if observation["test_setting"]["scenario_type"] == "loading":
            path_planned = self.get_main_road(observation)
            print('len(main_road)',len(path_planned))
            nearest_point, point_ind, min_distance = self.find_nearest_point(path_planned, goal)
            print("min_distance:",min_distance)
            print("nearest_point",nearest_point)
            dist = 0
            while point_ind:
                dist += math.sqrt((path_planned[point_ind][0] - path_planned[point_ind-1][0])**2 + (path_planned[point_ind][1] - path_planned[point_ind-1][1])**2)
                point_ind -= 1
                if dist > 30:
                    break
            path_planned = path_planned[0:point_ind+1]
            print("len(path_planned):",len(path_planned))
            keypoint_ind = len(path_planned)
            # print("keypoint:",keypoint)
            for i in range(len(path_planned)):
                path_planned[i]=path_planned[i][0:3]
                path_planned[i].append(True)
            handover = path_planned[-1]
            print("handover:",handover)
            astar_path = hybrid_a_star_planning(handover, goal, collision_lookup, observation, 2.0, 15,True)
            if astar_path is not None:
                print("len(astar_path):",len(astar_path.xlist))
                # print(astar_path.directionlist)
                for j in range(1,len(astar_path.xlist)):
                    path_planned.append([astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                    # if j+1 == len(astar_path.xlist):
                    #     keypoint += j-1
                    if astar_path.directionlist[j] == True and astar_path.directionlist[j+1] == False:
                        print("keypoint:",[astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                        keypoint_ind += j-1
                        #print("keypoint:",keypoint)
                print("keypoint_ind:",keypoint_ind)
                print("len(path):",len(path_planned))
                path_planned[keypoint_ind][-1]=2
                for k in range(len(path_planned)):
                    # print("k=",k)
                    if k < 50:
                        spd_planned.append(target_speed / 50 * k)
                    elif k >= 50 and k <= keypoint_ind-100:
                        spd_planned.append(target_speed)
                    elif (k > keypoint_ind-100) and (k <= keypoint_ind):
                        spd_planned.append(target_speed/ 100 * (keypoint_ind-k))
                    elif k > keypoint_ind and k <= keypoint_ind+100:
                        spd_planned.append(-target_speed_backward / 120 * (k-keypoint_ind+20))
                    elif k > keypoint_ind+100 and k < len(path_planned)-100:
                        spd_planned.append(-target_speed_backward)
                    elif k >= len(path_planned)-100:
                        spd_planned.append(-target_speed_backward / 100 * (len(path_planned)-k-1))
            else:
                return [],[]
        elif observation["test_setting"]["scenario_type"] == "unloading":
            # astar_star=[self._ego_x,self._ego_y,self._ego_yaw]
            path_planned = self.get_main_road(observation,False)
            astar_star = [observation['vehicle_info']['ego']['x'],
                          observation['vehicle_info']['ego']['y'],
                          observation['vehicle_info']['ego']['yaw_rad']]
            nearest_point, point_ind, min_distance = self.find_nearest_point(path_planned, astar_star)
            dist = 0
            while point_ind < len(path_planned):
                dist += math.sqrt((path_planned[point_ind][0] - path_planned[point_ind-1][0])**2 + (path_planned[point_ind][1] - path_planned[point_ind-1][1])**2)
                point_ind += 1
                # print(path_planned[point_ind+1][0])
                if dist > 20:
                    break
            print("min_distance:",min_distance)
            print("nearest_point",nearest_point)
            # print(path_planned)
            path_planned = path_planned[point_ind-1:]
            # print(path_planned)
            handover = path_planned[0]
            print("handover:",handover)
            astar_path = hybrid_a_star_planning(astar_star, handover, collision_lookup, observation, 2.0, 15,False)
            if astar_path is not None:
                print("len(astar_path):",len(astar_path.xlist))
                for i in range(len(path_planned)):
                    path_planned[i]=path_planned[i][0:3]
                    path_planned[i].append(True)
                path_planned1 = []
                for j in range(1,len(astar_path.xlist)):
                    path_planned1.append([astar_path.xlist[j],astar_path.ylist[j],astar_path.yawlist[j],astar_path.directionlist[j]])
                path_planned = path_planned1 + path_planned
                print("len(path):",len(path_planned))
                for k in range(len(path_planned)):
                    if k < 50:
                        spd_planned.append(target_speed / 50 * k)
                    else:
                        spd_planned.append(target_speed)
            else:
                return [],[]
        # print(path_planned)
        for i in range(1,len(path_planned)):
            path_planned[i].append(self.calculate_curvature(path_planned[i-1][0],path_planned[i-1][1],path_planned[i-1][2],
                                                            path_planned[i][0],path_planned[i][1],path_planned[i][2]))
        return path_planned[1:],spd_planned[1:]
        
        
    def get_main_road(self,observation,bo_in=True):
        """获取HD Map中ego车到达铲装平台的参考路径.     
        输入:observation——环境信息;
        输出:ego车到达目标区域的参考路径(拼接后).
        """
        main_road={"polygon-14":[["path-2","path-3","path-4","path-48","path-50","path-43","path-44"],
                                 ["path-13","path-14","path-58","path-31","path-37","path-38","path-39"]],
                   "polygon-27":[["path-36","path-5","path-6","path-7","path-8","path-9","path-70","path-71","path-72","path-73"],
                                 ["path-17","path-18","path-19","path-20","path-21","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-29":[["path-36","path-5","path-6","path-7","path-8","path-9","path-84","path-85","path-74","path-75"],
                                 ["path-22","path-23","path-59","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-25":[["path-36","path-5","path-6","path-7","path-8","path-9","path-68","path-69"],
                                 ["path-15","path-16","path-65","path-66","path-67","path-64","path-80","path-81"]],
                   "polygon-10":[["path-2","path-45","path-46","path-47","path-40","path-41"],
                                 ["path-77","path-10","path-11","path-27","path-28","path-25","path-26","path-39"]]}
        #################更新参数#################
        self._ego_x = observation['vehicle_info']['ego']['x']
        self._ego_y = observation['vehicle_info']['ego']['y']
        self._ego_v = observation['vehicle_info']['ego']['v_mps']
        self._ego_yaw = observation['vehicle_info']['ego']['yaw_rad']
        #################定位主车和目标点所在几何#################
        if bo_in:
            ego_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._ego_x,self._ego_y)
            ego_polygon_id = int( ego_polygon_token.split('-')[1])
            print("ego_polygon_token:",ego_polygon_token)
            goal_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._goal_x,self._goal_y)
            print("goal_polygon_token:",goal_polygon_token)
            ego_dubinspose_token = self.get_dubinspose_token_from_polygon\
                                (observation,(self._ego_x,self._ego_y,self._ego_yaw),ego_polygon_token)
            print("ego_dubinspose_token:",ego_dubinspose_token)
        else:
            ego_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._goal_x,self._goal_y)
            ego_polygon_id = int( ego_polygon_token.split('-')[1])
            print("ego_polygon_token:",ego_polygon_token)
            goal_polygon_token = observation['hdmaps_info']['tgsc_map'].get_polygon_token_using_node(self._ego_x,self._ego_y)
            print("goal_polygon_token:",goal_polygon_token)
            ego_dubinspose_token = self.get_dubinspose_token_from_polygon\
                                (observation,(self._goal_x,self._goal_y,self._goal_yaw),ego_polygon_token)
            print("ego_dubinspose_token:",ego_dubinspose_token)
        #################获取目标车最匹配的dubinspose#################
        
        # ego_dubinspose_id = int( ego_dubinspose_token.split('-')[1])
        # ego_dubinspose_token 作为 起点、终点的path拿到
        link_referencepath_tokens_ego_polygon = observation['hdmaps_info']['tgsc_map'].polygon[ego_polygon_id]['link_referencepath_tokens']
        # 去除掉 不包含 ego_dubinspose_token 的 path
        for _,path_token in enumerate(link_referencepath_tokens_ego_polygon):
            path_id = int( path_token.split('-')[1])
            link_dubinspose_tokens = observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['link_dubinspose_tokens']
            if ego_dubinspose_token not in link_dubinspose_tokens:
                pass
            else:
                only_one_path_token = path_token 
                only_one_path_id= path_id 
        print("path_token:",path_token)
        path_connected = []
        for road in main_road[goal_polygon_token]:
            if only_one_path_token in road:
                ind = road.index(only_one_path_token)
                terminal_path_id = int(road[ind].split('-')[1])
                terminal_path = observation['hdmaps_info']['tgsc_map'].reference_path[terminal_path_id]['waypoints']
                if bo_in:
                    _, nearest_point_ind, _ = self.find_nearest_point(terminal_path, [self._ego_x,self._ego_y])
                    terminal_path = terminal_path[nearest_point_ind:]
                    road_1 = road[ind+1:len(road)]
                    for road_segment in road_1:
                        path_id = int(road_segment.split('-')[1])
                        path_connected += observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['waypoints']
                    path_connected = terminal_path + path_connected
                else:
                    _, nearest_point_ind, _ = self.find_nearest_point(terminal_path, [self._goal_x,self._goal_y])
                    terminal_path = terminal_path[0:nearest_point_ind+1]
                    road_1 = road[:ind]
                    for road_segment in road_1:
                        path_id = int(road_segment.split('-')[1])
                        path_connected += observation['hdmaps_info']['tgsc_map'].reference_path[path_id]['waypoints']
                    path_connected =  path_connected + terminal_path
        return path_connected
    
    def get_dubinspose_token_from_polygon(self,observation,veh_pose:Tuple[float,float,float],polygon_token:str):
        id_polygon = int(polygon_token.split('-')[1])
        link_dubinspose_tokens = observation['hdmaps_info']['tgsc_map'].polygon[id_polygon]['link_dubinspose_tokens']
        dubinsposes_indicators = []
        for token in link_dubinspose_tokens:
            id_dubinspose = int(token.split('-')[1])
            dx = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['x'] - veh_pose[0]
            dy = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['y'] - veh_pose[1]
            dyaw = observation['hdmaps_info']['tgsc_map'].dubins_pose[id_dubinspose]['yaw'] - veh_pose[2]
            distance = math.sqrt(dx**2 + dy**2)
            if distance < 3 and abs(dyaw) < 1e-2:
                return token
    
    def line_equation(self,x, y, angle):
            if angle != np.pi / 2 and angle != -np.pi / 2:  # 避免斜率为无限大的情况
                m = np.tan(angle)
                b = y - m * x
                return m, b
            else:
                return float('inf'), x  # 对于垂直线，斜率为无限大，返回x作为截距

    def calculate_curvature(self,x1, y1, angle1, x2, y2, angle2):
        # 中点坐标
        # mid_x, mid_y = (x1 + x2) / 2.0, (y1 + y2) / 2.0

        # 如果两个方向角度相同，曲线不是圆弧，是直线
        if angle1 == angle2:
            return 0  # 直线的曲率为0，倒数为无穷大

        # 计算垂直于两点连线的直线的斜率（即中垂线）
        # 避免除以零的错误
        # if x2 != x1:
        #     slope_perpendicular = -(x2 - x1) / (y2 - y1)
        # else:
        #     slope_perpendicular = float('inf')
        
        # 使用点斜式方程计算两条直线的方程
        # y = mx + b，通过一个点和斜率求b
        # 计算两个方向的直线方程
        m1, b1 = self.line_equation(x1, y1, angle1)
        m2, b2 = self.line_equation(x2, y2, angle2)

        # 找到圆心（两条直线的交点）
        if m1 != float('inf') and m2 != float('inf'):
            cx = (b2 - b1) / (m1 - m2)
            cy = m1 * cx + b1
        elif m1 == float('inf'):
            cx = b1
            cy = m2 * cx + b2
        else:
            cx = b2
            cy = m1 * cx + b1

        # 计算圆心到任一点的距离（半径）
        radius = np.sqrt((cx - x1)**2 + (cy - y1)**2)

        # 计算曲率
        curvature = 1 / radius

        return curvature
        
    def find_nearest_point(self,path, point):
        """
        在路径上找到给定点的最近点。
        
        参数:
        - path: 路径点的列表，每个点是一个(x, y)元组。
        - point: 给定点，一个(x, y)元组。
        
        返回:
        - nearest_point: 路径上最近的点。
        - min_distance: 到给定点的最小距离。
        """
        
        # 初始化最小距离和最近点
        min_distance = float('inf')
        point_ind = 0
        # print(path)
        # 遍历路径上的每个点
        for i in range(len(path)):
            path_point = path[i]
            # 计算当前点与给定点之间的欧几里得距离
            distance = math.sqrt((path_point[0] - point[0])**2 + (path_point[1] - point[1])**2)
            
            # 更新最小距离和最近点
            if distance < min_distance:
                point_ind = i
                nearest_point = path_point
                min_distance = distance
        
        return nearest_point, point_ind,min_distance

         
              
