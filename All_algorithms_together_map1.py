# %%
from simpleai.search import SearchProblem, astar, breadth_first, depth_first, greedy, uniform_cost
import numpy as np
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.colors import LightSource
import plotly.graph_objects as px

# %%
mars_map = np.load("map.npy")
scale = 10.0177
nr, nc = mars_map.shape
max_height = 0.75

def xy_to_rc(x, y):
  r = int(nr - round(y/scale))
  c = int(round(x/scale))
  if 0 <= r < nr and 0 <= c < nc:
      return r,c
  else:
      return None

def get_xy(r,c):
  y = scale*(nr -r)
  x = scale*c
  return y,x

def rc_tuple_to_xy(px):
  r,c = xy_to_rc(px[0],px[1])

  return r,c


def get_height_rc(r,c):
  return mars_map[r][c]

# %%

initial_state = [
    (5000,7600) , (1400, 3600), (6050, 1750), (2500, 6000), (4300, 4700), (2300, 1370)
]

goal_state = [
    (3600,8600), (1900,5800), (4600,2500), (2200,9000), (730,9200), (4000,11500)
]


# %% [markdown]
# # A*

# %%
class AstarMarsRover(SearchProblem):
  def __init__(self, initial_state, goal_state):
        self.goal_state = goal_state
        super().__init__(initial_state) #Note: The stats will be in rows and columns 
  def actions(self,state):
      actual_altitud = get_height_rc(state[0], state[1])
      posible_actions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
      actions = []
      for dr, dc in posible_actions:
        if np.abs(actual_altitud - get_height_rc(state[0] + dr, state[1] + dc)) <= max_height:
          actions.append((dr, dc))

      return actions

  def result(self, state, action):
    return (state[0] + action[0], state[1] + action[1])


  def is_goal(self, state):
    return state == goal_state_px

  def cost(self, state, action, state2):
      return np.sqrt((action[0]**2) + (action[1]**2) + (get_height_rc(state[0], state[1]) - get_height_rc(state2[0], state2[1])))


  def heuristic(self,state):
    if state is None:
      return float('inf')
    return ((state[0] - goal_state_px[0])**2+(state[1] - goal_state_px[1])**2)

total_time = []
total_time = []
for i in range(6): #Loop for all 6 locations
    initial_state_px = rc_tuple_to_xy(initial_state[i])
    goal_state_px = rc_tuple_to_xy(goal_state[i])

    print(f"Initial state: {initial_state_px}")
    print(f"Goal state: {goal_state_px}")

    start_time = time.time()
    problem = AstarMarsRover(initial_state_px, goal_state_px)
    result = astar(problem, graph_search=True)
    time_path = time.time() - start_time
    total_time.append(time_path)
    if result != None:
      print(f"Path found for route {i+1}")
      print(f"The distance explored is: {result.cost} meters")
      print(f"Time for route number is: {i+1}: {time_path} seconds")
      path_px = [step[1] for step in result.path()]  

      path_x = []
      path_y = []
      path_z = []

      for r, c in path_px:
          y, x = get_xy(r, c)
          path_x.append(x)
          path_y.append(y)
          path_z.append(mars_map[r, c]) 


      x = scale * np.arange(mars_map.shape[1])
      y = scale * np.arange(mars_map.shape[0])
      X, Y = np.meshgrid(x, y)

      fig = px.Figure(data=[
          px.Surface(
              x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
              lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
              lightposition=dict(x=0, y=mars_map.shape[0] // 2, z=2 * mars_map.max())
          ),
          px.Scatter3d(
              x=path_x, y=path_y, z=path_z, name='Path', mode='markers+lines',
              marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Plotly3", size=4)
          )
      ],
          layout=px.Layout(
              scene_aspectmode='manual',
              scene_aspectratio=dict(x=1, y=mars_map.shape[0] / mars_map.shape[1], z=max(mars_map.max() / x.max(), 0.2)),
              scene_zaxis_range=[0, mars_map.max()]
          )
      )

      fig.show()
    else:
        print("No path found")

# %% [markdown]
# # DFS

# %%
initial_state_DFS = [
    (5000,7600) #It only ran for the first route.
]

goal_state_DFS = [
    (3600,8600)
]

# %%
class MarsRoverDFS(SearchProblem):
    def __init__(self, initial_state, goal_state):
        self.goal_state = goal_state
        super().__init__(initial_state) #Note: The stats will be in rows and columns 

    def actions(self, state):
        altitude_now = get_height_rc(state[0], state[1])
        posible_movements = [(-1, 0), (1, 0), (0, -1), (0, 1),  (-1, -1), (-1, 1), (1, -1), (1, 1)] #We included diagonal movements too
        actions = []
        for dr, dc in posible_movements:
            if np.abs(altitude_now - get_height_rc(state[0] + dr, state[1] + dc)) <= max_height: #Restriction implementation
                actions.append((dr, dc))
        return actions
    
    def result(self, state, action):
        return (state[0] + action[0], state[1] + action[1])
    
    def cost(self, state, action, state2):
        delta_h = np.abs(get_height_rc(state[0], state[1]) - get_height_rc(state2[0], state2[1])) #Change in height
        return np.sqrt(((action[0]*scale)**2)+((action[1]*scale)**2) + (delta_h**2)) #Euclidean distance in 3d in meters


    def is_goal(self, state):
        return state == self.goal_state

total_time = []
for i in range(len(initial_state_DFS)): #Loop for all 6 locations
    initial_state_px = rc_tuple_to_xy(initial_state_DFS[i])
    goal_state_px = rc_tuple_to_xy(goal_state_DFS[i])

    print(f"Initial state: {initial_state_px}")
    print(f"Goal state: {goal_state_px}")

    start_time = time.time()
    problem = MarsRoverDFS(initial_state_px, goal_state_px)
    result = depth_first(problem, graph_search=True)
    time_path = time.time() - start_time
    total_time.append(time_path)
    if result != None:
      print(f"Path found for route {i+1}")
      print(f"The distance explored is: {result.cost} meters")
      print(f"Time for route number is: {i+1}: {time_path} seconds")
      path_px = [step[1] for step in result.path()]  

      path_x = []
      path_y = []
      path_z = []

      for r, c in path_px:
          y, x = get_xy(r, c)
          path_x.append(x)
          path_y.append(y)
          path_z.append(mars_map[r, c]) 


      x = scale * np.arange(mars_map.shape[1])
      y = scale * np.arange(mars_map.shape[0])
      X, Y = np.meshgrid(x, y)

      fig = px.Figure(data=[
          px.Surface(
              x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
              lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
              lightposition=dict(x=0, y=mars_map.shape[0] // 2, z=2 * mars_map.max())
          ),
          px.Scatter3d(
              x=path_x, y=path_y, z=path_z, name='Path', mode='markers+lines',
              marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Plotly3", size=4)
          )
      ],
          layout=px.Layout(
              scene_aspectmode='manual',
              scene_aspectratio=dict(x=1, y=mars_map.shape[0] / mars_map.shape[1], z=max(mars_map.max() / x.max(), 0.2)),
              scene_zaxis_range=[0, mars_map.max()]
          )
      )

      fig.show()
    else:
        print("No path found")

# %% [markdown]
# # BFS

# %%
class MarsRoverBFS(SearchProblem):
    def __init__(self, initial_state, goal_state):
        self.goal_state = goal_state
        super().__init__(initial_state) #Note: The stats will be in rows and columns 

    def actions(self, state):
        altitude_now = get_height_rc(state[0], state[1])
        posible_movements = [(-1, 0), (1, 0), (0, -1), (0, 1),  (-1, -1), (-1, 1), (1, -1), (1, 1)] #We included diagonal movements too
        actions = []
        for dr, dc in posible_movements:
            if np.abs(altitude_now - get_height_rc(state[0] + dr, state[1] + dc)) <= max_height: #Restriction implementation
                actions.append((dr, dc))
        return actions
    
    def result(self, state, action):
        return (state[0] + action[0], state[1] + action[1])
    
    def cost(self, state, action, state2):
        delta_h = np.abs(get_height_rc(state[0], state[1]) - get_height_rc(state2[0], state2[1])) #Change in height
        return np.sqrt(((action[0]*scale)**2)+((action[1]*scale)**2) + (delta_h**2)) #Euclidean distance in 3d in meters


    def is_goal(self, state):
        return state == self.goal_state

total_time = []
for i in range(6): #Loop for all 6 locations
    initial_state_px = rc_tuple_to_xy(initial_state[i])
    goal_state_px = rc_tuple_to_xy(goal_state[i])

    print(f"Initial state: {initial_state_px}")
    print(f"Goal state: {goal_state_px}")

    start_time = time.time()
    problem = MarsRoverBFS(initial_state_px, goal_state_px)
    result = breadth_first(problem, graph_search=True)
    time_path = time.time() - start_time
    total_time.append(time_path)
    if result != None:
      print(f"Path found for route {i+1}")
      print(f"The distance explored is: {result.cost} meters")
      print(f"Time for route number is: {i+1}: {time_path} seconds")
      path_px = [step[1] for step in result.path()]  

      path_x = []
      path_y = []
      path_z = []

      for r, c in path_px:
          y, x = get_xy(r, c)
          path_x.append(x)
          path_y.append(y)
          path_z.append(mars_map[r, c]) 


      x = scale * np.arange(mars_map.shape[1])
      y = scale * np.arange(mars_map.shape[0])
      X, Y = np.meshgrid(x, y)

      fig = px.Figure(data=[
          px.Surface(
              x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
              lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
              lightposition=dict(x=0, y=mars_map.shape[0] // 2, z=2 * mars_map.max())
          ),
          px.Scatter3d(
              x=path_x, y=path_y, z=path_z, name='Path', mode='markers+lines',
              marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Plotly3", size=4)
          )
      ],
          layout=px.Layout(
              scene_aspectmode='manual',
              scene_aspectratio=dict(x=1, y=mars_map.shape[0] / mars_map.shape[1], z=max(mars_map.max() / x.max(), 0.2)),
              scene_zaxis_range=[0, mars_map.max()]
          )
      )

      fig.show()
    else:
        print("No path found")

# %% [markdown]
# # Greedy Search

# %%
class MarsRoverGreedy(SearchProblem):
    def __init__(self, initial_state, goal_state):
        self.goal_state = goal_state
        super().__init__(initial_state)

    def actions(self, state):
        altitude_now = get_height_rc(state[0], state[1])
        posible_movements = [
            (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        actions = []
        for dr, dc in posible_movements:
            if np.abs(altitude_now - get_height_rc(state[0] + dr, state[1] + dc)) <= max_height:
                actions.append((dr, dc))
        return actions

    def result(self, state, action):
        return (state[0] + action[0], state[1] + action[1])

    def cost(self, state, action, state2):
        delta_h = np.abs(get_height_rc(state[0], state[1]) - get_height_rc(state2[0], state2[1]))
        return np.sqrt(((action[0] * scale) ** 2) + ((action[1] * scale) ** 2) + (delta_h ** 2))

    def is_goal(self, state):
        return state == self.goal_state

    def heuristic(self, state):
        return np.linalg.norm(np.array(state) - np.array(self.goal_state))
total_time = []
for i in range(6): #Loop for all 6 locations
    initial_state_px = rc_tuple_to_xy(initial_state[i])
    goal_state_px = rc_tuple_to_xy(goal_state[i])

    print(f"Initial state: {initial_state_px}")
    print(f"Goal state: {goal_state_px}")

    start_time = time.time()
    problem = MarsRoverGreedy(initial_state_px, goal_state_px)
    result = greedy(problem, graph_search=True)
    time_path = time.time() - start_time
    total_time.append(time_path)
    if result != None:
      print(f"Path found for route {i+1}")
      print(f"The distance explored is: {result.cost} meters")
      print(f"Time for route number is: {i+1}: {time_path} seconds")
      path_px = [step[1] for step in result.path()]  

      path_x = []
      path_y = []
      path_z = []

      for r, c in path_px:
          y, x = get_xy(r, c)
          path_x.append(x)
          path_y.append(y)
          path_z.append(mars_map[r, c]) 


      x = scale * np.arange(mars_map.shape[1])
      y = scale * np.arange(mars_map.shape[0])
      X, Y = np.meshgrid(x, y)

      fig = px.Figure(data=[
          px.Surface(
              x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
              lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
              lightposition=dict(x=0, y=mars_map.shape[0] // 2, z=2 * mars_map.max())
          ),
          px.Scatter3d(
              x=path_x, y=path_y, z=path_z, name='Path', mode='markers+lines',
              marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Plotly3", size=4)
          )
      ],
          layout=px.Layout(
              scene_aspectmode='manual',
              scene_aspectratio=dict(x=1, y=mars_map.shape[0] / mars_map.shape[1], z=max(mars_map.max() / x.max(), 0.2)),
              scene_zaxis_range=[0, mars_map.max()]
          )
      )

      fig.show()
    else:
        print("No path found")
    

# %% [markdown]
# # Uniform Cost Search

# %%


class MarsRoverUCS(SearchProblem):
    def __init__(self, initial_state, goal_state):
        self.goal_state = goal_state
        super().__init__(initial_state)

    def actions(self, state):
        altitude_now = get_height_rc(state[0], state[1])
        possible_movements = [
            (-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)
        ]
        actions = []
        for dr, dc in possible_movements:
            new_r, new_c = state[0] + dr, state[1] + dc
            if 0 <= new_r < nr and 0 <= new_c < nc:
                if abs(altitude_now - get_height_rc(new_r, new_c)) <= max_height:
                    actions.append((dr, dc))
        return actions

    def result(self, state, action):
        return state[0] + action[0], state[1] + action[1]

    def cost(self, state, action, state2):
        delta_h = abs(get_height_rc(state[0], state[1]) - get_height_rc(state2[0], state2[1]))
        return np.sqrt((action[0] * scale) ** 2 + (action[1] * scale) ** 2 + delta_h ** 2)

    def is_goal(self, state):
        return state == self.goal_state



total_time = []
for i in range(len(initial_state)): #Loop for all 6 locations
    initial_state_px = rc_tuple_to_xy(initial_state[i])
    goal_state_px = rc_tuple_to_xy(goal_state[i])

    print(f"Initial state: {initial_state_px}")
    print(f"Goal state: {goal_state_px}")

    start_time = time.time()
    problem = MarsRoverUCS(initial_state_px, goal_state_px)
    result = uniform_cost(problem, graph_search=True)
    time_path = time.time() - start_time
    total_time.append(time_path)
    if result != None:
      print(f"Path found for route {i+1}")
      print(f"The distance explored is: {result.cost} meters")
      print(f"Time for route number is: {i+1}: {time_path} seconds")
      path_px = [step[1] for step in result.path()]  

      path_x = []
      path_y = []
      path_z = []

      for r, c in path_px:
          y, x = get_xy(r, c)
          path_x.append(x)
          path_y.append(y)
          path_z.append(mars_map[r, c]) 


      x = scale * np.arange(mars_map.shape[1])
      y = scale * np.arange(mars_map.shape[0])
      X, Y = np.meshgrid(x, y)

      fig = px.Figure(data=[
          px.Surface(
              x=X, y=Y, z=np.flipud(mars_map), colorscale='hot', cmin=0,
              lighting=dict(ambient=0.0, diffuse=0.8, fresnel=0.02, roughness=0.4, specular=0.2),
              lightposition=dict(x=0, y=mars_map.shape[0] // 2, z=2 * mars_map.max())
          ),
          px.Scatter3d(
              x=path_x, y=path_y, z=path_z, name='Path', mode='markers+lines',
              marker=dict(color=np.linspace(0, 1, len(path_x)), colorscale="Plotly3", size=4)
          )
      ],
          layout=px.Layout(
              scene_aspectmode='manual',
              scene_aspectratio=dict(x=1, y=mars_map.shape[0] / mars_map.shape[1], z=max(mars_map.max() / x.max(), 0.2)),
              scene_zaxis_range=[0, mars_map.max()]
          )
      )

      fig.show()
    else:
        print("No path found")



