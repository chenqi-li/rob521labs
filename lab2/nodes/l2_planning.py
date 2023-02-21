#!/usr/bin/env python3
#Standard Libraries
import numpy as np
import yaml
import pygame
import time
import pygame_utils
import matplotlib.image as mpimg
from skimage.draw import disk
from scipy.linalg import block_diag
import matplotlib.pyplot as plt

#Map Handling Functions
def load_map(filename):
    im = mpimg.imread("../maps/" + filename)
    im_np = np.array(im)  #Whitespace is true, black is false
    #im_np = np.logical_not(im_np)    
    return im_np

def load_map_yaml(filename):
    with open("../maps/" + filename, "r") as stream:
            map_settings_dict = yaml.safe_load(stream)
    return map_settings_dict

#Node for building a graph
class Node:
    def __init__(self, point, parent_id, cost):
        self.point = point.reshape(3,-1) # A 3 by 1 vector [x, y, theta]
        self.parent_id = parent_id # The parent node id that leads to this node (There should only every be one parent in RRT)
        self.cost = cost # The cost to come to this node
        self.children_ids = [] # The children node ids of this node
        return

#Path Planner 
class PathPlanner:
    #A path planner capable of perfomring RRT and RRT*
    def __init__(self, map_filename, map_setings_filename, goal_point, stopping_dist):
        #Get map information
        self.occupancy_map = load_map(map_filename)
        self.map_shape = self.occupancy_map.shape
        self.map_settings_dict = load_map_yaml(map_setings_filename)

        #Get the metric bounds of the map
        self.bounds = np.zeros([2,2]) #m
        self.bounds[0, 0] = self.map_settings_dict["origin"][0]
        self.bounds[1, 0] = self.map_settings_dict["origin"][1]
        self.bounds[0, 1] = self.map_settings_dict["origin"][0] + self.map_shape[1] * self.map_settings_dict["resolution"]
        self.bounds[1, 1] = self.map_settings_dict["origin"][1] + self.map_shape[0] * self.map_settings_dict["resolution"]

        #Robot information
        self.robot_radius = 0.22 #m
        self.vel_max = 0.4 #m/s (Feel free to change!)
        self.rot_vel_max = 1.6 #rad/s (Feel free to change!) # Chenqi: code works if rot_vel_max >= pi/2
        
        #Map information (for convenience)
        self.resolution = 0.05 #m/cell (Occupancy Grid Resolution)

        #Goal Parameters
        self.goal_point = goal_point #m
        self.stopping_dist = stopping_dist #m

        #Trajectory Simulation Parameters
        self.timestep = 1.0 #s
        self.num_substeps = 10

        #Planning storage
        self.nodes = [Node(np.zeros((3,1)), -1, 0)] # [Node(np.array([[2.8],[-2.8],[-np.pi/2]]), -1, 0)] #

        #RRT* Specific Parameters
        self.lebesgue_free = np.sum(self.occupancy_map) * self.map_settings_dict["resolution"] **2
        self.zeta_d = np.pi
        self.gamma_RRT_star = 2 * (1 + 1/2) ** (1/2) * (self.lebesgue_free / self.zeta_d) ** (1/2)
        self.gamma_RRT = self.gamma_RRT_star + .1
        self.epsilon = 2.5

        #Sampling parameter
        self.lowest_togo = np.inf
        
        #Pygame window for visualization
        self.window = pygame_utils.PygameWindow(
            "Path Planner", (800, 800), self.occupancy_map.shape, self.map_settings_dict, self.goal_point, self.stopping_dist)
        # print('wait for 1 sec')
        # time.sleep(1)
        return

    #Functions required for RRT
    def sample_map_space(self):
        #Return an [x,y] coordinate to drive the robot towards
        #print("TO DO: Sample point to drive towards")

        # Define the bounds to sample from
        largest_bounds = np.array([[0, 44], 
                      [-46, 10]])
        own_bounds = largest_bounds

        # Dynamic window to sample from, help reach goal in small room faster, remove for RSTAR if needed
        if self.goal_point[0] > 100 or self.goal_point[1] > 100:
            own_bounds = largest_bounds
        else:
            new_low_x = (self.goal_point[0] - self.lowest_togo*1.5)
            new_low_y = (self.goal_point[1] + self.lowest_togo*1.5)
            own_bounds = np.array([[max(largest_bounds[0,0], new_low_x), largest_bounds[0,1]],
                            [largest_bounds[1,0], min(largest_bounds[1,1], new_low_y)]])

        # Get the sampled point
        the_point = np.random.rand(2,1)
        the_point[0,0] = the_point[0,0]*(own_bounds[0,1] - own_bounds[0,0]) + own_bounds[0,0]
        the_point[1,0] = the_point[1,0]*(own_bounds[1,1] - own_bounds[1,0]) + own_bounds[1,0]

        # Visualizing the bounds in pygame
        self.window.add_point(own_bounds[:,0],radius = 5)
        self.window.add_point(own_bounds[:,1],radius = 5)

        return the_point
    
    def check_if_duplicate(self, point):
        #Check if point is a duplicate of an already existing node
        #print("TO DO: Check that nodes are not duplicates")

        for node in self.nodes:
            if np.linalg.norm(node.point[0:2] - point[0:2]) < 0.01:
                return True
        return False
    
    def closest_node(self, point):
        #Returns the index of the closest node
        #print("TO DO: Implement a method to get the closest node to a sapled point")
        
        #Convert Nodes to array
        node_xy_list = np.empty((2,0))
        for node in self.nodes:
            node_xy_list = np.hstack([node_xy_list, node.point[:2].reshape(2,-1)])
        return np.argmin(np.linalg.norm(node_xy_list - point, axis=0))
    
    def simulate_trajectory(self, node_i, point_s):
        #Simulates the non-holonomic motion of the robot.
        #This function drives the robot from node_i towards point_s. This function does has many solutions!
        #node_i is a 3 by 1 vector [x;y;theta] this can be used to construct the SE(2) matrix T_{OI} in course notation
        #point_s is the sampled point vector [x; y]
        # print("TO DO: Implment a method to simulate a trajectory given a sampled point")
        
        # Simulate the trajectory
        vel, rot_vel = self.robot_controller(node_i, point_s)
        robot_traj = self.trajectory_rollout(node_i, vel, rot_vel)

        return robot_traj
    
    def robot_controller(self, node_i, point_s):
        #This controller determines the velocities that will nominally move the robot from node i to node s
        #Max velocities should be enforced
        # print("TO DO: Implement a control scheme to drive you towards the sampled point")

        # Extract the variables
        x_i, y_i, theta_i = node_i
        x_s, y_s = point_s
        dx = x_s - x_i
        dy = y_s - y_i
        dist = np.sqrt(dx**2 + dy**2)

        # If too far away, move the target point to closer distance
        if dist > self.vel_max*self.timestep:
            dx = dx/dist*self.vel_max*self.timestep
            dy = dy/dist*self.vel_max*self.timestep
        dist = np.sqrt(dx**2 + dy**2)

        # Define the control law
        theta_s = np.arctan2(dy, dx)
        d_theta = theta_s - theta_i

        # If within [-pi/2, pi/2], move forward, otherwise, move backwards
        if d_theta > -np.pi/2 and d_theta < np.pi/2:
            linear_vel = dist/self.timestep
            angular_vel = d_theta/self.timestep
        elif d_theta <= -np.pi/2:
            linear_vel = -dist/self.timestep
            angular_vel = (d_theta+np.pi)/self.timestep
        elif d_theta >= np.pi/2:
            linear_vel = -dist/self.timestep
            angular_vel = (d_theta-np.pi)/self.timestep
        
        # If the change in angle is too large, reduce the linear velocity, so we can get closer to the point, otherwise it overshoots
        if abs(angular_vel) > np.pi/3:
            linear_vel *= 0.1

        # If angular velocity larger than max, set to max
        angular_vel = np.clip(angular_vel, -self.rot_vel_max, self.rot_vel_max)

        return linear_vel, angular_vel


    def trajectory_rollout(self, node_i, vel, rot_vel):
        # Given your chosen velocities determine the trajectory of the robot for your given timestep
        # The returned trajectory should be a series of points to check for collisions
        # print("TO DO: Implement a way to rollout the controls chosen")
        p = np.asarray([vel,rot_vel]).reshape(2,1)
        point_list = np.zeros([3, self.num_substeps+1])
        point_list[:, 0] = node_i.squeeze()
        substep_size = self.timestep / self.num_substeps #self.num_substeps / self.timestep

        for i in range(0, self.num_substeps):
            theta = point_list[2,i]
            G = np.asarray([[np.cos(theta), 0],
                            [np.sin(theta), 0],
                            [0, 1]])
            q_dot = np.matmul(G,p)
            point_list[:,i+1] = point_list[:,i] + q_dot.squeeze()*substep_size
        return point_list[0:3,1:]
    
    def point_to_cell(self, point):
        #Convert a series of [x,y] points in the map to the indices for the corresponding cell in the occupancy map
        #point is a 2 by N matrix of points of interest
        # print("TO DO: Implement a method to get the map cell the robot is currently occupying")

        # Convert from [x,y] points in map frame to occupancy_map pixel coordinates
        origin = np.asarray(self.map_settings_dict['origin'][:2]).reshape(2,-1)
        resolution = self.map_settings_dict['resolution']
        indices = (point-origin)/resolution # get indices
        # Occupancy map origin is top left, but map points origin bottom left
        indices[1,:] = self.map_shape[1]-indices[1,:] 

        return np.vstack([indices[1,:],indices[0,:]]).astype(int)

    def points_to_robot_circle(self, points):
        #Convert a series of [x,y] points to robot map footprints for collision detection
        #Hint: The disk function is included to help you with this function
        # print("TO DO: Implement a method to get the pixel locations of the robot path")

        centers = self.point_to_cell(points)
        radius = np.ceil(self.robot_radius/self.resolution).astype(int)
        occupancy_grid = np.zeros((self.map_shape))
        for i in range(centers.shape[1]):
            rr, cc = disk(centers[:,i],radius,shape=self.map_shape)
            occupancy_grid[rr,cc] = 1
        pixel_idxs = np.argwhere(occupancy_grid == 1).T
        return occupancy_grid, pixel_idxs

        # occupancy_grid is the np array of the occupancy grid with points on the robot path equal to one (assumes x-axis along rows and y-axis along columns)
        # pixel_idx is a 2xN np array of indices of the occupancy_grid that are equal to one (occupied)
    
    #Note: If you have correctly completed all previous functions, then you should be able to create a working RRT function

    #RRT* specific functions
    def ball_radius(self):
        #Close neighbor distance
        card_V = len(self.nodes)
        return min(self.gamma_RRT * (np.log(card_V) / card_V ) ** (1.0/2.0), self.epsilon)
    
    def connect_node_to_point(self, node_i, point_f):
        #Given two nodes find the non-holonomic path that connects them
        #Settings
        #node is a 3 by 1 node
        #point is a 2 by 1 point
        print("TO DO: Implement a way to connect two already existing nodes (for rewiring).")

        


        return np.zeros((3, self.num_substeps))
    
    def cost_to_come(self, trajectory_o):
        #The cost to get to a node from lavalle 
        print("TO DO: Implement a cost to come metric")
        return 0
    
    def update_children(self, node_id):
        #Given a node_id with a changed cost, update all connected nodes with the new cost
        print("TO DO: Update the costs of connected nodes after rewiring.")
        return

    #Planner Functions
    def rrt_planning(self):
        #This function performs RRT on the given map and robot
        #You do not need to demonstrate this function to the TAs, but it is left in for you to check your work
        while True: #Most likely need more iterations than this to complete the map!
            #Sample map space
            point = self.sample_map_space()

            #Get the closest point
            closest_node_id = self.closest_node(point)

            #Simulate driving the robot towards the closest point
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for collisions
            occupied_cells, occupied_idx = self.points_to_robot_circle(trajectory_o[0:2,:])

            # print(self.occupancy_map[occupied_idx[0,:].T, occupied_idx[1,:].T]) 
            if np.all(self.occupancy_map[occupied_idx[0,:].T, occupied_idx[1,:].T]):
                new_point = trajectory_o[:,-1]

                # Do not add if duplicate
                if self.check_if_duplicate(new_point): #continue if duplicate
                    continue

                # Do no add if too close to obstacles
                scale = 0.5
                traj1 = self.trajectory_rollout(new_point, self.vel_max*scale, 0)
                traj2 = self.trajectory_rollout(new_point, -self.vel_max*scale, 0)
                traj = np.hstack([traj1, traj2]) #, traj3, traj4, traj5, traj6])
                occupied_cells, occupied_idx = self.points_to_robot_circle(traj[0:2,:])
                if not np.all(self.occupancy_map[occupied_idx[0,:].T, occupied_idx[1,:].T]):
                    continue
                
                # Add node
                new_node = Node(new_point, closest_node_id, 0)
                self.nodes.append(new_node)
                
                # For visualizing in pygame
                self.window.add_point(np.copy(new_point[0:2]), radius=1, color=(0,0,255))

                # Checking if close to goal
                dist_to_goal = np.linalg.norm(self.goal_point - new_point[0:2].reshape(2,-1))
                if dist_to_goal < self.lowest_togo:
                    self.lowest_togo = dist_to_goal
                if dist_to_goal < self.stopping_dist:
                    print("Goal Reached!")
                    break
        return self.nodes
    
    def rrt_star_planning(self):
        #This function performs RRT* for the given map and robot        
        for i in range(1): #Most likely need more iterations than this to complete the map!
            #Sample
            point = self.sample_map_space()

            #Closest Node
            closest_node_id = self.closest_node(point)

            #Simulate trajectory
            trajectory_o = self.simulate_trajectory(self.nodes[closest_node_id].point, point)

            #Check for Collision
            print("TO DO: Check for collision.")

            #Last node rewire
            print("TO DO: Last node rewiring")

            #Close node rewire
            print("TO DO: Near point rewiring")

            #Check for early end
            print("TO DO: Check for early end")
        return self.nodes
    
    def recover_path(self, node_id = -1):
        path = [self.nodes[node_id].point]
        current_node_id = self.nodes[node_id].parent_id
        while current_node_id > -1:
            path.append(self.nodes[current_node_id].point)
            current_node_id = self.nodes[current_node_id].parent_id
        path.reverse()
        return path

def main():
    #Set map information
    map_filename = "willowgarageworld_05res.png"
    map_setings_filename = "willowgarageworld_05res.yaml"

    #robot information
    goal_point = np.array([[41.2], [-44.2]]) #m
    stopping_dist = 0.5 #m
    

    # Initialize Class
    path_planner = PathPlanner(map_filename, map_setings_filename, goal_point, stopping_dist)

    # #RRT Test
    start_time = time.time()
    nodes = path_planner.rrt_planning()
    end_time = time.time()
    print(f'Time to goal: {end_time-start_time}')
    node_path_metric = np.hstack(path_planner.recover_path())
    np.save("shortest_path.npy", node_path_metric)
    print(node_path_metric)
    for i in range(1, node_path_metric.shape[1]):
        path_planner.window.add_point(node_path_metric[:2,i], radius=1, color=(255, 0, 0))
        

    # #Simulate Trajectory Test
    # traj1 = path_planner.trajectory_rollout(np.array([0,0,0]),1,0.2)
    # traj2 = path_planner.trajectory_rollout(np.array([0,0,0]),0.01,1.5)
    # plt.scatter(traj1[0,:].T, traj1[1,:].T)
    # plt.scatter(traj2[0,:].T, traj2[1,:].T)
    # plt.show()

    # #Farther than we can reach
    # traj1 = path_planner.simulate_trajectory(np.array([0,0,0]),[1,10])
    # traj2 = path_planner.simulate_trajectory(np.array([0,0,0]),[1,-10])
    # traj3 = path_planner.simulate_trajectory(np.array([0,0,0]),[-1,-10])
    # traj4 = path_planner.simulate_trajectory(np.array([0,0,0]),[-1,10])
    # traj5 = path_planner.simulate_trajectory(np.array([0,0,0]),[10,10])
    # traj6 = path_planner.simulate_trajectory(np.array([0,0,0]),[10,-10])
    # traj7 = path_planner.simulate_trajectory(np.array([0,0,0]),[-10,-10])
    # traj8 = path_planner.simulate_trajectory(np.array([0,0,0]),[-10,10])
    # plt.scatter(traj1[0,:].T, traj1[1,:].T)
    # plt.scatter(traj2[0,:].T, traj2[1,:].T)
    # plt.scatter(traj3[0,:].T, traj3[1,:].T)
    # plt.scatter(traj4[0,:].T, traj4[1,:].T)
    # plt.scatter(traj5[0,:].T, traj5[1,:].T)
    # plt.scatter(traj6[0,:].T, traj6[1,:].T)
    # plt.scatter(traj7[0,:].T, traj7[1,:].T)
    # plt.scatter(traj8[0,:].T, traj8[1,:].T)
    # plt.show()

    # # Closer than we can reach
    # traj1 = path_planner.simulate_trajectory(np.array([0,0,0]),[0.01,0.1])
    # traj2 = path_planner.simulate_trajectory(np.array([0,0,0]),[0.01,-0.1])
    # traj3 = path_planner.simulate_trajectory(np.array([0,0,0]),[-0.01,-0.1])
    # traj4 = path_planner.simulate_trajectory(np.array([0,0,0]),[-0.01,0.1])
    # traj5 = path_planner.simulate_trajectory(np.array([0,0,0]),[0.1,0.1])
    # traj6 = path_planner.simulate_trajectory(np.array([0,0,0]),[0.1,-0.1])
    # traj7 = path_planner.simulate_trajectory(np.array([0,0,0]),[-0.1,-0.1])
    # traj8 = path_planner.simulate_trajectory(np.array([0,0,0]),[-0.1,0.1])
    # plt.scatter(traj1[0,:].T, traj1[1,:].T)
    # plt.scatter(traj2[0,:].T, traj2[1,:].T)
    # plt.scatter(traj3[0,:].T, traj3[1,:].T)
    # plt.scatter(traj4[0,:].T, traj4[1,:].T)
    # plt.scatter(traj5[0,:].T, traj5[1,:].T)
    # plt.scatter(traj6[0,:].T, traj6[1,:].T)
    # plt.scatter(traj7[0,:].T, traj7[1,:].T)
    # plt.scatter(traj8[0,:].T, traj8[1,:].T)
    # plt.show()

    # # Test point_to_cell
    # print("Chenqi: Test point_to_cell function")
    # point = np.array([[0, 0,  10, -10, -20],
    #                   [0, 10, 10, -10, -48.25]])
    # point = np.array([[-11.00, 44.00], 
    #                   [-44.25, 20.75]])
    # # print(point, '\n', path_planner.point_to_cell(point))
    # # path_planner.points_to_robot_circle(point)
    # for i in point.T:
    #     path_planner.window.add_point(i, radius=2, color=(0, 0, 255))
    #     print(path_planner)
    # plt.imshow(path_planner.occupancy_map)
    # plt.show()

    # Keep pygame running
    while True:
        pygame.display.update()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

if __name__ == '__main__':
    main()
