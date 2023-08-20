import os
import cv2
import pygame
import random
import numpy as np
class Tetris:
    def __init__(self, canvas_path, folder_path):
        self.canvasPath = canvas_path
        self.piecesRootFolder = folder_path
        self.fileNames = os.listdir(self.piecesRootFolder)
        if folder_path in self.canvasPath:
            self.fileNames.remove(canvas_path[len(folder_path)+1:])
            # print(fileNames)
        self.canvasImage_orig = cv2.imread(self.canvasPath)
        self.canvasImage_orig = cv2.resize(self.canvasImage_orig,
                                      (self.canvasImage_orig.shape[1] - 350, self.canvasImage_orig.shape[0] - 280),
                                      interpolation=cv2.INTER_AREA)
        self.validation_mat = [[0] * (self.canvasImage_orig.shape[1] + 2)] * (self.canvasImage_orig.shape[0] + 2)
        self.validation_mat = [
            [1 if i == 0 or i == len(self.validation_mat) - 1 or j == 0 or j == len(sublist) - 1 else element for j, element
             in
             enumerate(sublist)] for i, sublist in enumerate(self.validation_mat)]
        self.score=0

    def add_sounds(sound):
        sound_effect_file = r'C:Sounds\{}.mp3'.format(sound)
        sound_effect = pygame.mixer.Sound(sound_effect_file)
        sound_effect.play()

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

    def progress_of_game(self):
        game_over=False
        pygame.mixer.init()
        pygame.mixer.music.load(r'C:Sounds\high-voltage-background-action-music-for-pc-games-9593.mp3')
        pygame.mixer.music.play()
        pygame.mixer.music.set_volume(0.5)
        while pygame.mixer.music.get_busy():
            while not game_over:
                cv2.rectangle(self.canvasImage_orig, (20, 10), (200, 60), (0, 0, 0), cv2.FILLED)
                cv2.putText(self.canvasImage_orig, f"Points: {self.score}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (255, 255, 255),
                            2)
                current_img = random.choice(self.fileNames)
                img_path =self.piecesRootFolder+'\\'+current_img
                # filePaths_example = [os.path.join(piecesRootFolder, f_name) for f_name in fileNames ]
                pieceImage = cv2.imread(img_path)
                self.score += 10
                # cv2.imshow(window_name, image)
                # 1080,1920,3
                # print("pieceImage.shape: ", pieceImage.shape)
                random_number = random.randint(0, self.canvasImage_orig.shape[1] - pieceImage.shape[1])
                height, width, channels = pieceImage.shape
                # pieceLocation = np.array([0, int(canvasImage_orig.shape[1] / 2)])  # Top Left Corner #נקודת ההתחלה של התמונה
                pieceLocation = np.array([0, random_number])  # Top Left Corner #נקודת ההתחלה של התמונה
                pieceVelocity = np.array([1, 0])  # הכמות שהיא זזה בכל דפיקת שעון (משתנה לפי הלחיצות)
                isReachedEndOfCanvas = False
                add_sounds("cartoon-jump-6462")
                while not isReachedEndOfCanvas:
                    canvasImage = self.canvasImage_orig.copy()
                    canvasImage[pieceLocation[0]:pieceLocation[0] + height,
                    pieceLocation[1]:pieceLocation[1] + width, :] = pieceImage
                    cv2.imshow('canvas', canvasImage)
                    key = cv2.waitKey(1)
                    f = self.check_validation_in_matrix(self.validation_mat, pieceLocation[0], pieceLocation[0] + height,
                                                   pieceLocation[1],
                                                   pieceLocation[1] + width)
                    if f:
                        pieceLocation = pieceLocation + pieceVelocity
                    else:
                        isReachedEndOfCanvas = True
                    if key == ord('a') and pieceLocation[1] - 10 > 0:
                        if self.check_validation_in_matrix(self.validation_mat, pieceLocation[0], pieceLocation[0] + height,
                                                      pieceLocation[1] - 10,
                                                      pieceLocation[1] + width - 10):
                            pieceLocation[1] -= 10
                    elif key == ord('d') and pieceLocation[1] + 10 < canvasImage.shape[1] - width:
                        if self.check_validation_in_matrix(self.validation_mat, pieceLocation[0], pieceLocation[0] + height,
                                                      pieceLocation[1] + 10,
                                                      pieceLocation[1] + width + 10):
                            pieceLocation[1] += 10
                    elif key == ord('s'):
                        if self.check_validation_in_matrix(self.validation_mat, pieceLocation[0] + 3,
                                                      pieceLocation[0] + height + 3,
                                                      pieceLocation[1],
                                                      pieceLocation[1] + width):
                            pieceLocation[0] += 3
                self.add_sounds("shooting")
                pygame.time.wait(400)
                # isReachedEndOfCanvas = pieceLocation[0] + height > canvasImage.shape[0]
                self.update_matrix(self.validation_mat, pieceLocation[0], pieceLocation[0] + height, pieceLocation[1],
                              pieceLocation[1] + width)

                canvasImage_orig = canvasImage
                if pieceLocation[0] < 10:
                    break
            pygame.mixer.music.stop()
            self.add_sounds("failure-drum-sound-effect-2-7184")
            org = (300, 400)
            fontScale = 5
            color = (0, 0, 255)
            thickness = 15
            canvasImage = cv2.putText(canvasImage, 'Game Over', org, cv2.FONT_HERSHEY_SIMPLEX,
                                      fontScale, color, thickness, cv2.LINE_AA)
            cv2.imshow('canvas', canvasImage)
            # pygame.time.wait(4000)
            cv2.waitKey(2000)


t1 = Tetris(r'C:Images\Canvas.png', r'C:Images')
t1.progress_of_game()