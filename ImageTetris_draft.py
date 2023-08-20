import numpy as np
import cv2
import os
import random
import pygame


def add_sounds(sound):
    sound_effect_file = r'C:Sounds\{}.mp3'.format(sound)
    sound_effect = pygame.mixer.Sound(sound_effect_file)
    sound_effect.play()

def remove_background(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(image)
    cv2.drawContours(mask, contours, -1, (255, 255, 255), thickness=cv2.FILLED)
    result = cv2.bitwise_and(image, mask)
    return result


def background_opacity(fileNames):
    new_file_names=[]
    for img in fileNames:
        path = r'C:Images\{}'.format(img)
        piece = cv2.imread(path)
        tmp = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)
        _, alpha = cv2.threshold(tmp, 0, 255, cv2.THRESH_BINARY)
        b, g, r = cv2.split(piece)
        rgba = [b, g, r, alpha]
        dst = cv2.merge(rgba, 4)
        cv2.imwrite(r'C:Images\white{}'.format(img), dst)
        # fileNames.append("white{}".format(img))
        # fileNames.remove(img)
        new_file_names.append(r'white{}'.format(img))
        print(piece)
    return new_file_names


def check_validation_in_matrix(mat, height_start, height_end, width_start, width_end, right=False, left=False):
    if 1 in mat[height_end + 1][width_start + 1:width_end - 1]:
        return False
    # if 1 in [row[width_start + 1:width_end] for row in mat[height_start + 1:height_end + 1]]:  # or [row[width_start-1:width_end] for row in mat[height_start - 1:height_end]]:
    #     return False
    return True


def update_matrix(mat, height_start, height_end, width_start, width_end):
    for i in range(height_start, height_end):
        for j in range(width_start, width_end):
            mat[i][j] = 1


def show_score():
    canvas_width = 400
    canvas_height = 200
    canvas = np.ones((canvas_height, canvas_width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    font_thickness = 2
    score = 0
    score += 10
    canvas.fill(255)
    text = f"Score: {score}"
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_position = ((canvas_width - text_size[0]) // 2, (canvas_height + text_size[1]) // 2)
    cv2.putText(canvas, text, text_position, font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)


canvasPath = r'C:Images\Canvas.png'
piecesRootFolder = r'C:Images'
fileNames = os.listdir(piecesRootFolder)
fileNames.remove('Canvas.png')
# fileNames = background_opacity(fileNames)
print(fileNames)
canvasImage_orig = cv2.imread(canvasPath)
# cv2.namedWindow('tetris',cv2.WINDOW_NORMAL)
canvasImage_orig = cv2.resize(canvasImage_orig, (canvasImage_orig.shape[1] - 350, canvasImage_orig.shape[0] - 280),
                              interpolation=cv2.INTER_AREA)

validation_mat = [[0] * (canvasImage_orig.shape[1] + 2)] * (canvasImage_orig.shape[0] + 2)
validation_mat = [
    [1 if i == 0 or i == len(validation_mat) - 1 or j == 0 or j == len(sublist) - 1 else element for j, element in
     enumerate(sublist)] for i, sublist in enumerate(validation_mat)]
score = 0

game_over = False
pygame.mixer.init()
pygame.mixer.music.load(r'C:Sounds\high-voltage-background-action-music-for-pc-games-9593.mp3')
pygame.mixer.music.play()
pygame.mixer.music.set_volume(0.5)
while pygame.mixer.music.get_busy():
    while not game_over:
        cv2.rectangle(canvasImage_orig, (20, 10), (200, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(canvasImage_orig, f"Points: {score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                    2)
        current_img = random.choice(fileNames)
        img_path = r'C:Images\{}'.format(current_img)
        # filePaths_example = [os.path.join(piecesRootFolder, f_name) for f_name in fileNames ]
        pieceImage = cv2.imread(img_path)
        score+=10
        # cv2.imshow(window_name, image)
        # 1080,1920,3
        print("pieceImage.shape: ", pieceImage.shape)
        random_number = random.randint(0, canvasImage_orig.shape[1]-pieceImage.shape[1])
        height, width, channels = pieceImage.shape
        # pieceLocation = np.array([0, int(canvasImage_orig.shape[1] / 2)])  # Top Left Corner #נקודת ההתחלה של התמונה
        pieceLocation = np.array([0, random_number])  # Top Left Corner #נקודת ההתחלה של התמונה
        pieceVelocity = np.array([1, 0])  # הכמות שהיא זזה בכל דפיקת שעון (משתנה לפי הלחיצות)
        isReachedEndOfCanvas = False
        add_sounds("cartoon-jump-6462")
        while not isReachedEndOfCanvas:
            canvasImage = canvasImage_orig.copy()
            canvasImage[pieceLocation[0]:pieceLocation[0] + height,
            pieceLocation[1]:pieceLocation[1] + width, :] = pieceImage
            cv2.imshow('canvas', canvasImage)
            key = cv2.waitKey(1)
            f = check_validation_in_matrix(validation_mat, pieceLocation[0], pieceLocation[0] + height, pieceLocation[1],
                                           pieceLocation[1] + width)
            if f:
                pieceLocation = pieceLocation + pieceVelocity
            else:
                isReachedEndOfCanvas=True
            if key == ord('a') and pieceLocation[1] - 10 > 0:
                if check_validation_in_matrix(validation_mat, pieceLocation[0], pieceLocation[0] + height,
                                              pieceLocation[1] - 10,
                                              pieceLocation[1] + width - 10):
                    pieceLocation[1] -= 10
            elif key == ord('d') and pieceLocation[1] + 10 < canvasImage.shape[1] - width:
                if check_validation_in_matrix(validation_mat, pieceLocation[0], pieceLocation[0] + height,
                                              pieceLocation[1] + 10,
                                              pieceLocation[1] + width + 10):
                    pieceLocation[1] += 10
            elif key == ord('s'):
                if check_validation_in_matrix(validation_mat, pieceLocation[0] + 3, pieceLocation[0] + height + 3,
                                              pieceLocation[1],
                                              pieceLocation[1] + width):
                    pieceLocation[0] += 3
        add_sounds("shooting")
        pygame.time.wait(400)
            # isReachedEndOfCanvas = pieceLocation[0] + height > canvasImage.shape[0]
        update_matrix(validation_mat, pieceLocation[0], pieceLocation[0] + height, pieceLocation[1],
                      pieceLocation[1] + width)


        canvasImage_orig = canvasImage
        if pieceLocation[0] < 10:
            break
    pygame.mixer.music.stop()
    add_sounds("failure-drum-sound-effect-2-7184")
    org = (300, 400)
    fontScale = 5
    color = (0, 0, 255)
    thickness = 15
    canvasImage = cv2.putText(canvasImage, 'Game Over', org, cv2.FONT_HERSHEY_SIMPLEX,
                              fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('canvas', canvasImage)
    # pygame.time.wait(4000)
    cv2.waitKey(2000)





# pygame.mixer.init()
# music_file = "C:\\L5\\musicMk.mp3"
# pygame.mixer.music.load(music_file)

