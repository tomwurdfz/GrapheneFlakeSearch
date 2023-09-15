import onnxruntime as ort
import numpy as np
from PIL import Image, ImageChops
from split_image import reverse_split
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pyformulas as pf
import time
import pyautogui
import pygetwindow
import os
import tkinter
import keyboard
from tkinter import ttk



def _load_image(image_filename: str) -> Image.Image:
    """
    :param image_filename: filename of an image

    Load an image as a numpy array from the filename
    """
    return Image.open(image_filename)


def _resize(image: np.ndarray, size: tuple) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param size: tuple of length 2 representing the height and width of the picture

    Resize an image to a needed size
    """
    return cv2.resize(image, size)


def _normalize(image: np.ndarray) -> np.ndarray:
    """
    :param image: image represented as np.ndarray

    Normalize an image as if it's from ImageNet
    """
    IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
    IMAGENET_STD = np.array([0.229, 0.224, 0.225])
    image -= IMAGENET_MEAN[None, None, :]
    image /= IMAGENET_STD[None, None, :]
    return image


def _median_correction(image: np.ndarray, intensity=0.6) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param intensity: intensity of a median pixel in the resulting image

    Apply median correction
    """
    r, g, b = image[:, :, 0], image[:, :, 1], image[:, :, 2]
    rmedian, gmedian, bmedian = np.median(r), np.median(g), np.median(b)
    result = image
    result[:, :, 0] *= intensity / rmedian
    result[:, :, 1] *= intensity / gmedian
    result[:, :, 2] *= intensity / bmedian
    return np.clip(result, 0., 1.)


def _prepare_image(image: Image.Image, size: tuple) -> np.ndarray:
    """
    :param image: image represented as np.ndarray
    :param size: a tuple representing the size of the resulting picture

    Prepare an image for fitting to the model
    """
    # get numpy array from the image
    image = np.array(image).astype(np.float32) / 255.

    # run resize and normalization
    return _normalize(_resize(_median_correction(image), size)).transpose((2, 0, 1))[None, :, :, :]


def _get_pred(model: ort.InferenceSession, image: np.ndarray, input_name: str = 'input') -> np.ndarray:
    """
    :param model: onnx inference session
    :param image: image represented as np.ndarray
    :param input_name: the name of the input of onnx model (default: 'input')

    Get a prediction for an image
    """
    return model.run(None, {input_name: image})[1]


def _pred2segm(pred: np.ndarray, classes_ids: np.ndarray = np.arange(4), colors: np.ndarray = None) -> np.ndarray:
    """
    :param pred: prediction got from onnx inference session
    :param classes_ids: indices of classes
    :param colors: colors corresponding to classes. np.ndarray of a shape (n_classes, 3), where the last dimension specifies color of a class

    Transfers model prediction to segmentation picture
    """
    if colors is None:
        # colors = np.array([[227, 217, 207],  # Background
        #                    [176, 179, 162],  # Monolayer
        #                    [109, 121, 117],  # Bilayer
        #                    [63, 56, 46]])  # Three layer
        colors = np.array([[255, 255, 255],  # Background
                           [0, 0, 0],  # Monolayer
                           [0, 0, 0],  # Bilayer
                           [128, 128, 255]])  # Three layer

    # create segmentation from predictions
    pred = pred.argmax(axis=1, keepdims=True)
    segm = np.zeros((pred.shape[0], 3, pred.shape[2], pred.shape[3]))
    for ind in classes_ids:
        segm += (pred == ind) * colors[ind][None, :, None, None]


    return segm.transpose((0, 2, 3, 1))


def load_model(model_filename: str) -> ort.InferenceSession:
    """
    :param model_filename: filename of onnx model

    Create an onnx inference session from the filename
    """
    execution_provider_priority_list = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    return ort.InferenceSession(model_filename, providers=execution_provider_priority_list)


def get_segmentation(model: ort.InferenceSession, image_filename: str, size: tuple = (720, 512),
                     colors: np.ndarray = None, multiplied: bool = False) -> np.ndarray:
    """
    :param model: onnx inference session
    :param image_filename: a filename for an image
    :param size: a size of the picture that model accepts as an input
    :param colors: colors of classes. np.ndarray for a shape (n_classes, 3), where the last dimension specifies color
    :param multiplied: if True return multiplied initial image with the segmentation, otherwise return just the segmentation

    Get segmentation from an onnx inference session and an image filename
    """
    image = _load_image(image_filename)
    initial_size = (image.width, image.height)

    prepared_image = _prepare_image(image, size)
    pred = _get_pred(model, prepared_image)

    segm = _pred2segm(pred, colors=colors).squeeze()
    resized_segm = cv2.resize(segm, initial_size)

    # apply multiplication if needed
    result = resized_segm.astype(np.uint8)
    if multiplied:
        result = np.array(ImageChops.multiply(image, Image.fromarray(result)))

    return result.astype(np.uint8)

def crop_and_split(file_directory, picture_directory):
    """
    :param file_directory: directory of original file
    :param picture_directory: directory of splitted images
    :return: the number of subimages per row and column

    Not used in automatic detection, split images to 500x500 subimages
    """
    global image
    image = Image.open(file_directory)
    length = image.size[0]
    width = image.size[1]
    new_length = length - length%500
    new_width = width - width%500
    image = image.crop((int((length%500)/2), int((width%500)/2),
                    int(new_length + (length%500)/2), int(new_width + (width%500)/2)))
    columns = int(new_length/500)
    rows = int(new_width/500)
    for row in range(0, rows):
        for column in range(0,columns):

            file_name = (picture_directory +str(row*columns+column)+ '.jpg')
            print((row*columns+column+1),'/',(rows*columns))
            new_image = image
            new_image = image.crop((500*column,500*row,500*(column + 1),500*(row+1)))
            new_image = new_image.save(file_name, quality=100, subsampling=0)
    return rows, columns


def determinesize(im):

    """

    :param im: Image file of the inferenced graphene image
    :return: True if a graphene large enough is detected

    the "1500" here represents the number of pixels for a genphene to count
    can be easliy reset

    Determines if flakes within the image are large enough
    """
    counter = 0
    for pixel in im.getdata():
        if pixel == (0,0,0):
            counter += 1
    sizelimit = False
    if counter >= 1500:
        sizelimit = True
    print(counter)
    return sizelimit

def alert(image, sizelimit):

    """
    :param image: Image file of the inferenced graphene image
    :param sizelimit: Trye if a graphene large enough is detected within the image
    :return: image with blue edges if large flakes are detected

    Give a visual alert if large flakes are detected
    """
    if sizelimit == True:
        rows = len(image)
        columns = len(image[0])
        for i in range (0, 30):
            for j in range(0, columns):
                image[i,j] = (255,0,0)
    return image



def inference(counter):

    """

    :param counter: the number of image taken in order, starting from 1, only used for naming

    Saves the output of the model
    """
    input_image = 'C:/Users/Kostya/Desktop/Flake Detector/screenshot_'+str(counter)+'.jpg'
    multiplied_segm = get_segmentation(onnx_model, input_image, multiplied=True)
    multiplied_output_filename = 'C:/Users/Kostya/Desktop/Flake Detector/overlay_'+str(counter)+'.jpg'
    Image.fromarray(multiplied_segm).save(multiplied_output_filename)


def getcoordinate(time, xdimension, ydimension):
    """

    :param time: time elapsed since the beginning of the start
    :param xdimension: length of the x dimension
    :param ydimension: length of the y dimension
    :return: the x and y coordinate of a flake
    """
    
    xdim = xdimension
    ydim = ydimension
    
    xtime = round(time % (xdim*5+0.4), 1)
    xcoord = 0
    ycoord = 0
    ycoord = round(time // (xdim*2.5 + 0.2), 0)
    if (0 <= xtime and xtime <= (xdim*2.5 + 0.2)):
        xcoord = round(xtime // 0.4, 0)
    elif ((xdim*2.5 + 0.2) < xtime):
        xcoord = (xdim*6.25) - round((xtime - (xdim*2.5 + 0.2)) // 0.4, 0)
    xcoord = round(xcoord*0.16, 2)
    ycoord = round(ycoord*0.16, 2)
    return (xcoord, ycoord)


if __name__ == '__main__':
    print("Using", ort.get_device())
    #User should minimize the Python Shell and put the Window of Interest on the front

    counter=1
    counter2=1

    #Operate is a global variable used to record the user's operation from the GUI
    global operate
    operate = " "

    #Global variable for scanning region, 8 is the default value, possible values are 8, 16, 24 and 32
    global xdimension
    xdimension = 8

    global ydimension
    ydimension = 8 
    
    #Directory for the screenshots
    file_directory = 'C:/Users/Kostya/Desktop/Flake Detector/screenshot_'

    #Get all windows currently opened
    titles = pygetwindow.getAllTitles()

    #specify the name of the Window of Interest
    window = pygetwindow.getWindowsWithTitle('NIS-Elements BR')[0]

    #Specify the region of the Window
    left = 30
    top = 120
    right = 1430
    bottom = 990

    #A bitmap display to display image output
    fig = plt.figure()
    canvas = np.zeros((480,640))
    screen = pf.screen(canvas)

    #File name of the inference model
    onnx_filename = 'C:/Users/Kostya/Desktop/Flake Detector/ocrnet_hrnet_alldata720x512.onnx'

    #Running status, True = inference running
    status = True

    #List of timestamp for images with flakes detected
    timelist = []

    #Initialization of the start time variable
    autoscan_time = time.time()

    # load the model
    onnx_model = load_model(onnx_filename)
    pyautogui.confirm("Confirm if ready")

    #Variable for pause, True = Paused
    pause = False

    #Variable for manual saving, True = Image will be saved
    save = False

    #The GUI for user controls
    win = tkinter.Tk()
    win.geometry("440x160")
    button_dict={}
    option= ["Start Scanning", "Stop Scanning", "Pause", "Save Image", "Set to 08x08mm", "Set to 08x16mm",
             "Set to 08x24mm", "Set to 08x32mm","Set to 16x08mm", "Set to 16x16mm",
             "Set to 16x24mm", "Set to 16x32mm","Set to 24x08mm", "Set to 24x16mm",
             "Set to 24x24mm", "Set to 24x32mm","Set to 32x08mm", "Set to 32x16mm",
             "Set to 32x24mm", "Set to 32x32mm",]

    #Command behind each button
    counter4 = 1
    counter5 = 1
    
    for i in option:

        def func(x=i):
            global operate
            operate = x
        button_dict[i]=ttk.Button(win, text=i, command= func)
        button_dict[i].place(x = int(counter4*100 - 80), y = int(counter5*30-20))
        counter4 = counter4 + 1

        if counter4 == 5:
            counter4 = 1
            counter5 = counter5 + 1



    while status == True:
        #If in paused state
        if pause:
            time.sleep(0.5)
            if keyboard.is_pressed("p") or operate == "Pause":
                pause = False
                print("Resume")
            operate = " "

        #If not in paused state
        else:

            #Inference session
            ort_session = ort.InferenceSession(onnx_filename, providers=["CUDAExecutionProvider"])

            #Reset the variables before each loop
            sizelimit = False
            save = False

            #Takes a screenshot and crop the image
            im = pyautogui.screenshot()
            im = im.crop((left+8, top+1, right-8, bottom-8))

            #Start time for each loop
            start_time = time.time()
            file_name = (file_directory +str(counter)+'.jpg')

            #Saves image with full quality
            im = im.save(file_name, quality=100, subsampling=0)

            #Sart time for each inference
            start_time_2 = time.time()
            inference(counter)

            #Time spent during ingerence
            print('%d --- inference time %s seconds ---' % (counter, round(time.time() - start_time_2,4)))
            img = mpimg.imread('C:/Users/Kostya/Desktop/Flake Detector/overlay_'+str(counter)+'.jpg')
            im2 = Image.open('C:/Users/Kostya/Desktop/Flake Detector/overlay_'+str(counter)+'.jpg')

            #Determine if flakes of significant size is detected
            sizelimit = determinesize(im2)

            #If so, save the raw image and overlay image in the flakes folder and update the flake image counter
            if sizelimit == True:
                im = Image.open('C:/Users/Kostya/Desktop/Flake Detector/screenshot_'+str(counter)+'.jpg')
                im = im.save('C:/Users/Kostya/Desktop/Flake Detector/Flakes/'+str(counter2)+'.jpg', quality=100, subsampling=0)
                im2 = im2.save('C:/Users/Kostya/Desktop/Flake Detector/Flakes/'+str(counter2)+'Overlay.jpg', quality=100, subsampling=0)
                print("Image ", counter2, " Saved", round(time.time() - autoscan_time,2))

                #Append the time since the beginning of physical scanning to the list
                timelist.append(round(time.time() - autoscan_time,2))
                counter2 = counter2 + 1

            #Convert BGR image to RGB image so the displayed color is correct
            new_img = np.zeros((861,1384,3))
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            RGB_img = alert(RGB_img,sizelimit)

            #Update the display with new image

            screen.update(RGB_img)

            #Shows the time spend since the start of the loop
            print('%d --- total time %s seconds ---' % (counter, round(time.time() - start_time,4)))

            #Update the GUI
            win.update()

            #Change variables according to user input, keyboard or GUI.
            if keyboard.is_pressed("s"):
                save = True

            if save == False and operate != "Save Image":
                os.remove('C:/Users/Kostya/Desktop/Flake Detector/overlay_'+str(counter)+'.jpg')
                os.remove(file_directory +str(counter)+'.jpg')

            if save == True or operate == "Save Image":
                print("Image Manually Saved")
                
            if operate[-1] == "m":
                if operate[7] == "0":
                    xdimension = 8
                if operate[7] == "1":
                    xdimension = 16
                if operate[7] == "2":
                    xdimension = 24
                if operate[7] == "3":
                    xdimension = 32
                if operate[10] == "0":
                    ydimension = 8
                if operate[10] == "1":
                    ydimension = 16
                if operate[10] == "2":
                    ydimension = 24
                if operate[10] == "3":
                    ydimension = 32
                print("Dimension", operate)

            if operate == "Set to 08x16mm":
                xdimension = 8
                ydimension = 16
                print("Dimension", operate)

            if keyboard.is_pressed("p") or operate == "Pause":
                pause = True
                print("Paused")
            if keyboard.is_pressed("1") or operate == "Start Scanning":
                print("Started Automatic Scanning")

                #record the time when physical scanning starts
                autoscan_time = time.time()

                #clear the list of time when images with detected flakes are recorded
                timelist = []

            if keyboard.is_pressed("2") or operate == "Stop Scanning":
                print("Stopped Automatic Scanning")

                #Stop the scan and quit the loop
                status = False

            #Reset the operate variable used to record user input from GUI
            operate = " "

            #Update the total images counter
            counter+=1

    #Summary of scan
    print("Captured ",len(timelist), " images automatically")
    print("The coordinates are")

    #Counter3 is only used as a temporary variable to generate index of
    counter3 = 1

    #Write the txt file
    #Write the header
    with open('report.txt', 'w') as f:
        f.write("Captured " + str(len(timelist)) + " images automatically \n \n")

    #Write coordinate for each automatically saved image with flakes
    for element in timelist:
        print("Image ", counter3, ": ", getcoordinate(element, xdimension, ydimension), " mm")
        with open('report.txt', 'a') as f:
            f.write("Image "+ str(counter3)+  ": "+ str(getcoordinate(element, xdimension, ydimension))+ "mm \n")
        counter3 = counter3 + 1

    time.sleep(10000)
