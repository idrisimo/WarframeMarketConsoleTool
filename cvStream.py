
import cv2
from pytesseract import pytesseract
# import pyautogui
from PIL import ImageGrab
import numpy as np
import time
import win32gui
import win32ui
import win32con
from pytesseract import pytesseract
from pytesseract import Output
from handlers import format_pytesseract_as_data, get_brackets, get_market_price
import pandas as pd

def list_window_names():
    def winEnumHandler(hwnd, ctx):
        if win32gui.IsWindowVisible(hwnd):
            print("Window stuff: ",hex(hwnd), win32gui.GetWindowText(hwnd), hwnd)
    win32gui.EnumWindows(winEnumHandler, None)
def window_capture():

    # define window size
    # w = 1920  # set this
    # h = 1080  # set this


    # bit file save location
    bmpfilenamename = "outbitfile.bmp"  # set this

    # select window
    hwnd = win32gui.FindWindow(None, "Streamlabs Desktop")
    #hwnd = None

    if hwnd != None:
        window_rect = win32gui.GetWindowRect(hwnd)
        w = window_rect[2] - window_rect[0]
        h = window_rect[3] - window_rect[1]
    else:
        w = 1920
        h = 1080

    # get the window image data
    wDC = win32gui.GetWindowDC(hwnd)
    dcObj = win32ui.CreateDCFromHandle(wDC)
    cDC = dcObj.CreateCompatibleDC()
    dataBitMap = win32ui.CreateBitmap()
    dataBitMap.CreateCompatibleBitmap(dcObj, w, h)
    cDC.SelectObject(dataBitMap)
    cDC.BitBlt((0, 0), (w, h), dcObj, (500, 500), win32con.SRCCOPY)

    # save the screenshot
    # dataBitMap.SaveBitmapFile(cDC, bmpfilenamename)
    signedIntsArray = dataBitMap.GetBitmapBits(True)
    img = np.frombuffer(signedIntsArray, dtype='uint8')
    img.shape = (h, w, 4)


    # Free Resources
    dcObj.DeleteDC()
    cDC.DeleteDC()
    win32gui.ReleaseDC(hwnd, wDC)
    win32gui.DeleteObject(dataBitMap.GetHandle())

    img = img[...,:3]
    img = np.ascontiguousarray(img)
    return img


pytesseract.tesseract_cmd = "E:\\Tesseract-OCR\\tesseract.exe"
loop_time = time.time()
df = pd.DataFrame(columns=['text', 'average_plat'])
while True:

    screenshot = ImageGrab.grab(bbox=(90, 690, 780, 980))  # Location on screen of screenshot
    screenshot = np.array(screenshot)
    # screenshot = cv2.cvtColor(screenshot,cv2.COLOR_RGB2BGR) # Normal colour
    screenshot = cv2.cvtColor(screenshot,cv2.COLOR_RGB2GRAY)
    screenshot = cv2.threshold(screenshot, 80,255, cv2.THRESH_BINARY)[1]
    # Returns text as sentences in a list
    # words_in_image = pytesseract.image_to_string(screenshot)
    # print(words_in_image)
    # words_in_image_cleaned = [words for words in words_in_image.split("\n") if words != '']
    # cv2.putText(screenshot, 'word', (90, 260 - 16),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)
    # print(words_in_image_cleaned)

    # Image processing
    image_data = pytesseract.image_to_data(screenshot, output_type=Output.DICT)
    formatted_data = format_pytesseract_as_data(image_data)

    

    for line in formatted_data:
        x, y, text = line['left'], line['top'], line['text']
        # print(text)
        brackets = get_brackets(text)
        plat_list = []
        for item in brackets:
            if 'Prime' in item:
                # Adding a dataframe stops too many calls from happening (hopefully)
                if item not in df["text"].values: 
                    average_plat = str(get_market_price(item))
                    print('----------------appending to df-------------------')
                    df = df.append({'text': item, 'average_plat': average_plat}, ignore_index=True)
                    
                else:
                    print('----------------getting data from df-------------------')
                    average_plat = df.loc[df['text'] == item, 'average_plat'].item()
                    print(f'Item: {item} Average Plat: {average_plat}')   
                plat_list.append(average_plat)
                # print(plat_list)
        # print(df)
        cv2.putText(screenshot, ' '.join(plat_list), (x + 500, y -30),cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    # for i, word in enumerate(image_data['text']):
    #     if word != "" or word != "\n" or word != " ":
    #         # print(repr(word))
    #         x, y, w, h = image_data['left'][i], image_data['top'][i], image_data['width'][i], image_data['height'][i]
    #         cv2.rectangle(screenshot, (x, y), (x + w, y + h), (0, 0, 255), 1)
    #         cv2.putText(screenshot, 'word', (x, y - 16), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 1)



    img = cv2.imshow("computer vision", screenshot)
    print(f'FPS: {1 / (time.time() - loop_time)}')
    loop_time = time.time()
    time.sleep(0.05)  # delay loop
    if cv2.waitKey(1) == ord('q'):
        cv2.destroyAllWindows()
        break

print("Done")
