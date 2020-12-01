import pygame
import os


data_dir = r'D:\dataset\chinese_charecters'

font_dir = r'C:\Windows\Fonts'
font_paths = ['simkai.ttf', 'simsun.ttc']

# 0x4e00 - 0x9fa5 chinese characters encoding range

start, end = 0x4e00, 0x9fa5
pygame.init()

for each_font in font_paths:
    if 'kai' in each_font:
        path = os.path.join(data_dir, 'kai', 'images')
        os.makedirs(path)
        font = pygame.font.Font(os.path.join(font_dir, each_font), 32)
        for word_idx in range(start, end):
            word = chr(word_idx)
            rtext = font.render(word, True, (255, 255, 255), (0, 0, 0))
            pygame.image.save(rtext, os.path.join(path, f'{word_idx}.png'))
    elif 'sun' in each_font:
        path = os.path.join(data_dir, 'sun', 'images')
        os.makedirs(path)
        font = pygame.font.Font(os.path.join(font_dir, each_font), 32)
        for word_idx in range(start, end):
            word = chr(word_idx)
            rtext = font.render(word, True, (255, 255, 255), (0, 0, 0))
            pygame.image.save(rtext, os.path.join(path, f'{word_idx}.png'))


