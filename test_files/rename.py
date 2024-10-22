import os
import time
FIRST_NUM = 101
DIR_PATH = r"D:\Downloads\oil_datasets"


while True:
    # if detect same file name, rename file
    if os.path.exists(DIR_PATH + "\\image_cut1.png"):
        # rename file
        os.rename(DIR_PATH + "\\image_cut1.png", DIR_PATH + "\\image" + str(FIRST_NUM) + ".png")
        print("Rename file to image" + str(FIRST_NUM) + ".png")
        FIRST_NUM += 1
    time.sleep(5)