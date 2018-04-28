import numpy as np 
import pygame
import cv2
import random
import pygame
 
class Computer():
        def __init__(self):
                self.padWid, self.padHei = 8, 100
                self.x, self.y = 0, screen_height/2 - self.padHei/2
                self.speed = 4
                
                self.score = 0
       
        def scoring(self):
                
                if self.score == 10:
                        print ("player 1 wins!")
                        exit()
       
        def movement(self):
            if self.y + (self.padHei/2) > ball.y:
                self.y = np.maximum(0,self.y - self.speed)
            elif self.y + (self.padHei/2) <ball.y:
                self.y = np.minimum(self.y + self.speed,screen_height - self.padHei)
       
        def draw(self):
                pygame.draw.rect(screen, (255, 255, 255), (self.x, self.y, self.padWid, self.padHei))
 
class Player():
        def __init__(self):
                self.padWid, self.padHei = 8, 100
                self.x, self.y = screen_width-8, screen_height/2 - self.padHei/2
                self.speed = 4
                
                self.score = 0
       
        def scoring(self):
                
                if self.score == 10:
                        print ("Player 2 wins!")
                        exit()
       
        def movement(self,action):
                
                if action[1]==1:
                        self.y -= self.speed
                elif action[2]==1:
                        self.y += self.speed
       
                if self.y <= 0:
                        self.y = 0
                elif self.y >= screen_height-self.padHei:
                        self.y = screen_height-self.padHei
       
        def draw(self):
                pygame.draw.rect(screen, (255, 255, 255), (self.x, self.y, self.padWid, self.padHei))
 
class Ball():
        def __init__(self):
                self.x, self.y = screen_width/2, screen_height/2
                self.speed_x = 5*random.choice([random.uniform(0.5,1),-random.uniform(0.5,1)])
                self.speed_y = 6*random.choice([-1,1])
                self.size =16  
       
        def movement(self):
            self.x += self.speed_x
            self.y += self.speed_y
            reward=0
            terminal=False
            #wall col
            if self.y <= 0:
                    self.speed_y *= -1
            elif self.y >= screen_height-self.size:
                    self.speed_y *= -1

            if self.x <= 0:
                check = 0
                for n in range(-self.size, computer.padHei):
                    if self.y == computer.y + n:
                        self.speed_x *= -1
                        check = 1
                if check == 0:
                    self.__init__()
                    player.__init__()
                    computer.__init__()
                    player.score += 1
                    reward=1
                    terminal=True

            elif self.x >= screen_width-self.size-player.padWid:
                check = 0
                for n in range(-self.size, player.padHei):
                    if self.y == player.y + n:
                        self.speed_x *= -1
                        check = 1
                if check == 0:
                    # print(self.x)
                    self.__init__()
                    player.__init__()
                    computer.__init__()
                    # self.speed_x = 4
                    computer.score += 1
                    reward=-1
                    terminal=True
            
            return reward,terminal
        def draw(self):
                pygame.draw.rect(screen, (255, 255, 255), (self.x, self.y, self.size, self.size))

def get_present_screen():
    pygame.event.pump()
    screen.fill((0, 0, 0))
    ball.draw()
    player.draw()
    player.scoring()
    computer.draw()
    computer.scoring()
    pygame.display.flip()
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    return image_data
def get_next_screen(action,freq):
    # pygame.event.pump()
    # print('hi')
    screen.fill((0, 0, 0))
    freq += action
    player.movement(action)
    reward,terminal=ball.movement()
    computer.movement()
    ball.draw()
    player.draw()
    computer.draw()
    pygame.display.flip()
    image_data = pygame.surfarray.array3d(pygame.display.get_surface())
    return reward,image_data,terminal, freq


screen_size=(640,480)
screen_height=480
screen_width=640
screen = pygame.display.set_mode(screen_size)
pygame.display.set_caption("Ping Pong")
pygame.font.init()
clock = pygame.time.Clock()
FPS = 60
action=np.zeros(3)
ball = Ball()
player = Player()
computer = Computer()
i=get_present_screen()



