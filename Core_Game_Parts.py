import pygame
import numpy as np

# global constants

SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
BG_COLOR = (120, 120, 120)
DRAW_COLOR = (50, 50, 50)

DRAW_RADIUS = 2
ERASE_RADIUS = 5  

CAR_WIDTH, CAR_HEIGHT = 20, 40
DEFAULT_START_X, DEFAULT_START_Y = 900, 426
DEFAULT_START_ANGLE = -45
DEFAULT_START_SPEED = 0
ACCELERATION = 0.05
BRAKE_FORCE = 0.1
MAX_SPEED = 5.00
FRICTION = 0.025
MIN_TURN_ANGLE = 1.5
MAX_TURN_ANGLE = 2

TRACK_SAVE_PATH = "monza_draw.png"
CAR_IMAGE_PATH = "Track_images/car.png"
TRACK_IMAGE_PATH = r"Track_images\track1.png"

finish_line_rect=pygame.Rect(DEFAULT_START_X+30,DEFAULT_START_Y-20,10,100)
checkpoint_data=[
    (834,520,10,120,0),
    (600,540,10,120,0),
    (110,569,10,120,90),
    (285,483,10,120,0),
    (366,314,10,120,0),
    (355,173,10,120,0),
    (450,109,10,120,0),
    (606,170,10,120,0),
    (818,91,10,120,0),
    (1127,88,10,120,310),
    (1094,270,10,120,0),
    (920,346,10,120,45)
]

#car class 
class Car:
    """
    Class representing a car in the racing simulation.
    """
    def __init__(self, image_path, x, y, angle=0, speed=0):
        self.original_image = pygame.transform.scale(
            pygame.image.load(image_path).convert_alpha(), 
            (CAR_WIDTH, CAR_HEIGHT)
        )
        image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(image, (CAR_WIDTH, CAR_HEIGHT))
        self.x, self.y = x, y
        self.angle = angle
        self.speed = speed
        self.rect=self.image.get_rect(center=(self.x,self.y))
        self.mask=pygame.mask.from_surface(self.image)

    def move(self):
        rad = np.radians(self.angle)
        self.x += self.speed * np.cos(rad)
        self.y -= self.speed * np.sin(rad)
        self.rect.center = (self.x, self.y)

    def draw(self, screen):
        self.image = pygame.transform.rotate(self.original_image, self.angle)
        self.rect = self.image.get_rect(center=(self.x, self.y))
        self.mask = pygame.mask.from_surface(self.image)
        screen.blit(self.image, self.rect.topleft)
        
    def get_rect(self):
        return self.rect
    
# Ray casting function
def ray_casting(car, track_surface):
    """Cast rays from the car's position to detect the track boundaries.

    Args:
        car (class): The car object.
        track_surface (Surface): The surface of the track.

    Returns:
        sensor_distance (float): The distance to the nearest track boundary.
        sensor_endpoint (tuple): The (x, y) coordinates of the sensor endpoint.
    """
    sensor_distance = []
    sensor_endpoint = []
    sensor_angle = [-45, 0, 45]

    for angle in sensor_angle:
        ray_angle = car.angle + angle
        ray_x, ray_y = car.x, car.y
        distance = 0
        max_distance = 200

        while distance < max_distance:
            rad = np.radians(ray_angle)
            ray_x += np.cos(rad)
            ray_y -= np.sin(rad)
            distance += 1

            if not (0 <= ray_x < SCREEN_WIDTH and 0 <= ray_y < SCREEN_HEIGHT):
                break

            pixel_color = track_surface.get_at((int(ray_x), int(ray_y)))[0:3]
            if pixel_color == DRAW_COLOR:
                break

        sensor_distance.append(distance)
        sensor_endpoint.append((ray_x, ray_y))

    return sensor_distance, sensor_endpoint