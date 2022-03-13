from __future__ import annotations
from cmath import e, sqrt
from enum import Enum
from time import sleep
from typing import Dict, List
from queue import PriorityQueue
import pygame
import random
import cv2
from PIL import Image
import numpy as np

# speed_per_level = {
#     0 : 0.48 , 1 : 0.43, 2 : 0.38, 3 : 0.33, 4 : 0.28, 5 : 0.23, 6 : 0.18, 7 : 0.13,
#     8 : 0.8, 9 : 0.6, 10 : 0.5, 11 : 0.5, 12 : 0.5, 13 : 0.4, 14 : 0.4, 15 : 0.4,
#     16 : 0.3, 17 : 0.3, 18 : 0.3, 19 : 0.2, 20 : 0.2, 21 : 0.2, 22 : 0.2, 23 : 0.2,
#     24 : 0.2, 25 : 0.2, 26 : 0.2, 27 : 0.2, 28 : 0.2, 29 : 0.1,
# }

s_width = 800
s_height = 700
play_width = 300
play_height = 600
block_size = 30
 
top_left_x = (s_width - play_width) // 2
top_left_y = s_height - play_height
 
S = [['.34','12.'],['1.','23','.4']]
Z = [['12.','.34'],['.1','32','4.']] 
I = [['1','2','3','4'],['1234']]
O = [['12','43']]
J = [['1..','234'],['21','3.','4.'],['432','..1'],['.4','.3','12']]
L = [['..1','432'],['4.','3.','21'],['234','1..',],['12','.3','.4']]
T = [['.1.','234'],['2.','31','4.'],['432','.1.'],['.4','13','.2']]
 
shapes = [S, Z, J, I, O, L, T]
shape_colors = [(0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0), (255, 165, 0), (0, 0, 255), (128, 0, 128)]

class Move(Enum):
    LEFT = 0
    RIGHT = 1
    DOWN = 2
    ROTATE = 3

class Piece(object):
    def __init__(self,x,y,rotation,shape):
        self.x = x
        self.y = y
        self.shape = shape
        self.color = shape_colors[shapes.index(shape)]
        self.rotation = rotation
    def copy(self) -> Piece:
        return Piece(self.x,self.y,self.rotation,self.shape)
    pass
 
def create_grid(locked_postions = {}):
    grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]
    for (j,i) in locked_postions: 
        grid[i][j] = locked_postions[(j,i)]
    return grid
    
def convert_shape_format(current_piece):
    positions = []
    format = current_piece.shape[current_piece.rotation % len(current_piece.shape)]
    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column != '.':
                positions.append((current_piece.x + j,current_piece.y + i))
    return positions
 
def valid_space(shape, grid):
    accepted_pos = [[(j,i) for j in range(10) if grid[i][j] == (0,0,0)] for i in range(20)]
    accepted_pos = [j for sub in accepted_pos for j in  sub]
    formatted = convert_shape_format(shape)
    for pos in formatted: 
        if( (pos not in accepted_pos) and pos[1] > -1) : 
            return False
    return True 
  
def check_lost(positions):
    for pos in positions: 
        _, y = pos
        if y < 0:
            return True
    return False
 
def get_shape():
    return Piece (4,-2,0,random.choice(shapes))
 
def draw_grid(surface,grid):
    for i in range(len(grid)):
        pygame.draw.line(surface,(120,120,120),(top_left_x,top_left_y + i*block_size),(top_left_x + play_width,top_left_y + i*block_size),)
        for j in range(len(grid[1])):
            pygame.draw.line(surface,(120,120,120),(top_left_x + j*block_size,top_left_y),(top_left_x + j*block_size,top_left_y + play_height),)

def can_move_segment(grid,segment : Dict):
    for (j,i) in segment:
        if i + 1 == len(grid) or (grid[i+1][j] != (0,0,0) and (j,i+1) not in segment):
            return False
    return True

def move_segment_by_one(grid : List,segment : Dict):
    sorted_keys = sorted(segment.keys(),key= lambda tup : tup[1],reverse = True)
    new_segment = {}
    for (j,i) in sorted_keys:
        new_segment[(j,i+1)] = segment[(j,i)]
    for (j,i) in new_segment:
        grid[i][j] = new_segment[(j,i)]
    for (j,i) in segment:
        if not (j,i) in new_segment: grid[i][j] = (0,0,0)
    return grid,new_segment

def drop_segments(grid:List, segments:List):
    found = True
    while found:
        found = False
        for i in range(len(segments)):
            if can_move_segment(grid,segments[i]):
                found = True
                grid, segments[i] = move_segment_by_one(grid,segments[i])               
    return grid

def get_segment(grid,row,column,segment : Dict):
    list = [-1,1]
    segment[(column,row)] = grid[row][column]
    for i in list:
        if row + i >= 0 and row + i != len(grid):
            if grid[row + i][column] != (0,0,0):
                if segment.get((column,row +i),None) == None: 
                    segment = get_segment(grid,row + i,column,segment)
                else : continue
    for j in list:
        if column + j >= 0 and column + j != len(grid[row]):
            if grid[row][column + j] != (0,0,0):
                if segment.get((column + j,row),None) == None:    
                    segment = get_segment(grid,row,column+j,segment)
                else : continue
    return segment
    
def get_segments(grid: List,row):
    if row < 0 : return []
    segments = []
    for column in range(len(grid[row])):
        if grid[row][column] == (0,0,0): continue
        found = False
        for seg in segments: 
            if (column,row) in seg:
                found = True
                break
        if found : continue
        segment = get_segment(grid,row,column,{})
        segments.append(segment)
    return segments

def delete_row(grid,row):
    for column in range(len(grid[row])):
        grid[row][column] = (0,0,0) 
    segments = get_segments(grid, row-1)
    grid = drop_segments(grid,segments)  
    return grid

def clear_rows(grid,locked_positions):
    final_reward = 0
    sequence_rows = 0
    current_locked = {}
    for (j,i) in locked_positions:
        current_locked[(j,i)] = locked_positions[(j,i)]
    previous_locked = {}
    stop = False   
    while not stop:

        stop = True
        for (j,i) in current_locked:
            if previous_locked.get((j,i),None) == None or previous_locked.get((j,i),None) != current_locked[(j,i)]:
                stop = False
                break
        
        index = len(grid)-1
        previous_row = False  
        while index>0:

            next = False
            for element in grid[index]:
                if element == (0,0,0):
                    next = True
                    index -= 1
                    break
            if next:
                if previous_row:
                    final_reward += pow(sequence_rows,2)
                    sequence_rows = 0
                previous_row = False
                continue
            
            grid = delete_row(grid,index)
            sequence_rows += 1
            previous_row = True

            index -= 1

        for (j,i) in current_locked:
            previous_locked[(j,i)] = current_locked[(j,i)]
        current_locked = {}
        
        for i in range(len(grid)):
            for j in range(len(grid[i])):
                if not grid[i][j] == (0,0,0) : current_locked[(j,i)] = grid[i][j]
    return final_reward,grid,current_locked

def draw_next_shape(shape, surface):
    font = pygame.font.SysFont('comicsans', 30)
    label = font.render('Next Shape', 1, (255,255,255))

    sx = top_left_x + play_width + 50
    sy = top_left_y + play_height /2

    format = shape.shape[shape.rotation % len(shape.shape)]

    for i, line in enumerate(format):
        row = list(line)
        for j, column in enumerate(row):
            if column != '.':
                pygame.draw.rect(surface,shape.color,(sx + j * block_size ,sy + i * block_size,block_size,block_size),0)
    surface.blit(label,(sx ,sy - 50))

def draw_window(surface,grid,next_piece):
    surface.fill((0,0,0))
    pygame.font.init()
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            pygame.draw.rect(surface,grid[i][j],(top_left_x + j * block_size ,top_left_y + i * block_size,block_size,block_size),0)
    pygame.draw.rect(surface,(255,0,0), (top_left_x,top_left_y,play_width,play_height),4)
    draw_grid(surface,grid)
    draw_next_shape(next_piece,surface)
    pygame.display.update()
    pass
 
def main():
    grid  = create_grid()
    change_piece = False
    score = 0 
    level = 0
    run = True
    current_piece = get_shape()
    next_piece = get_shape()
    locked_positions = {}
    # clock = pygame.time.Clock()
    fall_time = 0

    while run:
        grid = create_grid(locked_positions)
        possible_grids = get_all_possible_moves(grid,current_piece)
        render(grid)        
        
        i = random.randrange(len(possible_grids))
        tuple = possible_grids[i]
        locked_positions = tuple[1]
        if(check_lost(locked_positions)):
            run = False
            pygame.quit()
        current_piece = next_piece
        next_piece = get_shape()

        # fall_time += clock.get_rawtime()
        # clock.tick()
        # lev = speed_per_level.get(level,None)
        # if (lev == None) : lev = 29
        # if fall_time/1000 > speed_per_level.get(lev):
        #     fall_time = 0
        #     current_piece.y += 1
        #     if not (valid_space(current_piece, grid)):
        #         current_piece.y-=1
        #         change_piece = True

        # for event in pygame.event.get():
        #     if event.type == pygame.QUIT:
        #         run = False
        #     if event.type == pygame.KEYDOWN:
        #         if event.key == pygame.K_LEFT:
        #             current_piece.x -= 1
        #             if not valid_space(current_piece,grid):
        #                 current_piece.x +=1
        #         if event.key == pygame.K_RIGHT:
        #             current_piece.x += 1
        #             if not valid_space(current_piece,grid):
        #                 current_piece.x -=1
        #         if event.key == pygame.K_DOWN:
        #             current_piece.y += 1
        #             if not valid_space(current_piece,grid):
        #                 current_piece.y -=1
        #         if event.key == pygame.K_UP:
        #             current_piece.rotation += 1
        #             if not valid_space(current_piece,grid):
        #                 current_piece.rotation -= 1

        # shape_pos = convert_shape_format(current_piece)
        # for i in range(len(shape_pos)):
        #     x, y = shape_pos[i]
        #     if y>-1:
        #         grid[y][x] = current_piece.color
        
        # if change_piece:
        #     for i in range(len(shape_pos)):
        #         x, y = shape_pos[i]
        #         locked_positions[(x,y)] = current_piece.color            
        #     current_piece = next_piece
        #     next_piece = get_shape()
        #     increase,grid,locked_positions = clear_rows(grid,locked_positions)
        #     if increase != 0:
        #         score += increase
        #         if score % 10 ==0:
        #             level += 1
        #     change_piece = False
        # draw_window(win,grid,next_piece)

        # if check_lost(locked_positions):
        #     run = False
        #     pygame.display.quit()

def get_grid_height(grid : List):
    height = 1000
    for column in range(len(grid[0])):
        current_height = 0
        row  = 0
        while row != len(grid)-1 and grid[row + 1][column] == (0,0,0) and current_height != height:
            current_height +=1
            row += 1
        if current_height < height:
                height = current_height
    return height

def get_heighest_empty_segment(grid,height):
    final_segment = {}
    segments = []
    heighest_segment = 1000
    for row in range(len(grid)-1,height-1,-1):
        for column in range(len(grid[row])):
            if grid[row][column] != (0,0,0): continue
            found = False
            for seg in segments: 
                if (column,row) in seg:
                    found = True
                    break
            if found : continue
            segment = get_empty_segment(grid,row,column,{},height)
            segments.append(segment)
            segment_keys = sorted(segment.keys(),key= lambda tup : tup[1])
            if(segment_keys[0][1]<heighest_segment):
                heighest_segment = segment_keys[0][1]
                final_segment = segment
    return final_segment

def get_empty_segment(grid,row,column,segment : Dict,height):
    list = [-1,1]
    if (row == len(grid)-1): segment[(column,row)] = 1
    else : segment[(column,row)] = 0 
    for i in list:
        if row + i >= height and row + i != len(grid):
            if grid[row + i][column] == (0,0,0):
                if segment.get((column,row +i),None) == None: 
                    segment = get_empty_segment(grid,row + i,column,segment,height)
                else : continue
            elif(i == 1):
                segment[(column,row)] = 1
    for j in list:
        if column + j >= 0 and column + j != len(grid[row]):
            if grid[row][column + j] == (0,0,0):
                if segment.get((column + j,row),None) == None:    
                    segment = get_empty_segment(grid,row,column+j,segment,height)
                else : continue
    return segment

def get_all_possible_moves(grid : List,current_piece : Piece):
    height = get_grid_height(grid)
    segment = get_heighest_empty_segment(grid,height)
    all_final_states = []
    segment = sorted(segment.items(),key= lambda tup : tup[1],reverse = True)
    
    checked = []
    
    for ((column,row),edge) in segment:
        if edge == 0:
            break
        paths,checked = find_paths(grid,current_piece,(column,row),checked)
        if paths != None:
            all_final_states.extend(paths)
    return all_final_states

def find_paths(grid : List,current_piece : Piece, pixel,checked):
    final_paths = []

    target_x = pixel[0]
    target_y = pixel[1]
    
    path_found = False
    priority_queue = PriorityQueue()
    moves = []

    temp_piece = current_piece.copy()

    for rotation in range(len(current_piece.shape)): 
        shape_index = rotation % len(current_piece.shape)

        for i in range(len(temp_piece.shape[shape_index])-1,-1,-1):
            for j in range(len((temp_piece.shape[shape_index])[i])-1,-1,-1):
                if((temp_piece.shape[shape_index])[i][j] != '.'):
                    x_diff, y_diff = j , i
                    temp_piece.rotation = rotation
                    temp_piece.x = target_x - x_diff
                    temp_piece.y = target_y - y_diff

                    if(check_variation((temp_piece.x,temp_piece.y,abs(temp_piece.rotation % len(temp_piece.shape))),checked)):
                        continue

                    checked.append((temp_piece.x,temp_piece.y,abs(temp_piece.rotation % len(temp_piece.shape))))
                    
                    if(not valid_space(temp_piece,grid)):
                        continue

                    target_piece = temp_piece.copy()
                    stored_grid = copy_grid(grid)
                    stored_locked = {}

                    for row in range(len(stored_grid)):
                        for column in range(len(stored_grid[row])):
                            if (stored_grid[row][column]!=(0,0,0)):
                                stored_locked[(column,row)] = stored_grid[row][column]

                    shape_pos = convert_shape_format(target_piece)

                    for _,(x,y) in enumerate(shape_pos):  
                        stored_locked[(x,y)] = target_piece.color
                        if y > -1:
                            stored_grid[y][x] =  target_piece.color

                    temp_piece = current_piece.copy()
                    priority = round(measure_shape_distance(temp_piece,target_piece),5)
                    item = (moves.copy(),temp_piece.copy())
                    priority_queue.put((priority,random.randrange(0,10000000,1),item))

                    visited = []
                    while not priority_queue.qsize() == 0 and len(visited) <= 90:
                        entry = priority_queue.get()
                        state = entry[2]
                        moves = state[0]
                        temp_piece = state[1]

                        if (temp_piece.x + x_diff == target_x and temp_piece.y + y_diff == target_y):
                            path_found = True
                            break 

                        if(check_variation((temp_piece.x,temp_piece.y,abs(temp_piece.rotation % len(temp_piece.shape))),visited)):
                            continue

                        visited.append((temp_piece.x,temp_piece.y,abs(temp_piece.rotation % len(temp_piece.shape))))
                        
                        temp_piece.x -= 1
                        if(valid_space(temp_piece,grid)):
                            moves.append(Move.LEFT)
                            priority = round(measure_shape_distance(temp_piece,target_piece),5)
                            item = (moves.copy(),temp_piece.copy())
                            priority_queue.put((priority,random.randrange(0,1000000),item))
                            moves.pop()
                        temp_piece.x += 1

                        temp_piece.x += 1
                        if(valid_space(temp_piece,grid)):
                            moves.append(Move.RIGHT)
                            priority = round(measure_shape_distance(temp_piece,target_piece),5)
                            item = (moves.copy(),temp_piece.copy())
                            priority_queue.put((priority,random.randrange(0,1000000),item))                
                            moves.pop()
                        temp_piece.x -= 1

                        temp_piece.y += 1
                        if(valid_space(temp_piece,grid)):
                            moves.append(Move.DOWN)
                            priority = round(measure_shape_distance(temp_piece,target_piece),5)
                            item = (moves.copy(),temp_piece.copy())
                            priority_queue.put((priority,random.randrange(0,1000000),item))
                            moves.pop()
                        temp_piece.y -= 1

                        temp_piece.rotation += 1
                        if(valid_space(temp_piece,grid)):
                            moves.append(Move.ROTATE)
                            priority = round(measure_shape_distance(temp_piece,target_piece),5)
                            item = (moves.copy(),temp_piece.copy())
                            priority_queue.put((priority,random.randrange(0,1000000),item))
                            moves.pop()
                        temp_piece.rotation -= 1

                    if(path_found):
                        final_paths.append((stored_grid,stored_locked,moves))
    
    return final_paths,checked

def copy_grid(grid):
    new_grid = [[(0,0,0) for _ in range(10)] for _ in range(20)]
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            new_grid[i][j] = grid[i][j]   
    return new_grid

def measure_shape_distance(current : Piece,target : Piece):
    current_x = current.x
    current_y = current.y
    current_format = current.shape[current.rotation % len(current.shape)]

    target_x = target.x
    target_y = target.y
    target_format = target.shape[target.rotation % len(target.shape)]

    final_distance = 0
    map = []

    for i in range(len(current_format)):
        for j in range(len(current_format[i])):
            temp = current_format[i][j]
            found = False
            if(temp !=  '.'):
                for t_i in range(len(target_format)):
                    for t_j in range(len(target_format[t_i])):
                        if(target_format[t_i][t_j] == temp):
                            pixel1 = (current_x + j,current_y + i)
                            pixel2 = (target_x + t_j,target_y + t_i)
                            map.append((pixel1,pixel2))
                            break
                    if found: 
                        break

    for i in range(len(map)):
        tuple = map[i]
        pixel1 = tuple[0]
        pixel2 = tuple[1] 
        final_distance += measure_pixel_distance(pixel1,pixel2)
    
    return final_distance/len(map)

def measure_pixel_distance(pixel1,pixel2):
    return sqrt(pow(pixel1[0]-pixel2[0],2) + pow(pixel1[1]-pixel2[1],2)).real

def check_variation(variation,visited):
    for element in visited:
        if( element[0] == variation[0] and element[1] == variation[1] and element[2] == variation[2]):
            return True
    return False

def reset():
    return create_grid({})

def make_move(possible_moves,action):
    tuple = possible_moves[action]
    grid = tuple[0]
    locked_positions = tuple[1]
    done = False
    if(check_lost(locked_positions)):
        done = True
    if not done: reward,grid,locked_positions = clear_rows(grid,locked_positions)
    else: reward = -50
    return grid, reward, done

def render(grid):
    rgb,bin = get_image(grid)
    rgb = cv2.resize(rgb,(200,400),interpolation=cv2.INTER_AREA)
    cv2.imshow("image", rgb) 
    cv2.waitKey(500)
    cv2.destroyAllWindows()

def get_image(grid):
    processed_grid = np.zeros((20,10,3), dtype=np.uint8)
    for i in range(len(grid)):
        for j in range(len(grid[i])):
            processed_grid[i][j] = grid[i][j]
    rgb_img = Image.fromarray(processed_grid, 'RGB')
    rgb_img = np.asarray(rgb_img)
    rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)	
    gray_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
    _,binary_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY)
    return rgb_img,binary_img

# win = pygame.display.set_mode((s_width,s_height))
# pygame.display.set_caption('Tetris')
# main()