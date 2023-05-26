from django.shortcuts import redirect,render
from .models import Input_Image, Processed_Image
from django.core.files.images import ImageFile
from django.contrib.auth.models import User
from PIL import Image
from .forms import ImageForm
from django.contrib import messages
from django.contrib.auth import login,authenticate, logout
import cv2
import re
import numpy as np
from rembg import remove
import time
import sys
import os


def home(request):
    return render(request, "ourapp/register.html")

def signup(request):
    
    if request.method == "POST":
        nusername = request.POST['username']
        fname = request.POST['fname']
        lname = request.POST['lname']
        nemail = request.POST['email']
        pass2 = request.POST['pass2']
        pass1 = request.POST['pass1']
        mobno = request.POST['mobno']
        
        if User.objects.filter(username = nusername).exists() :
            messages.error(request,"Username already  Exists")
        elif  User.objects.filter(email = nemail).exists():
            messages.error(request,"Email is already register")
        elif pass1 != pass2 :
            messages.error(request,"Confirmed Password did not match the entered Password")
        elif (len(pass1) < 8):
            messages.error(request,"Password should contain atleast 8 characters")
        elif not re.search("[a-z]", pass1):
            messages.error(request,"Password should contain atleast one Lowercase letter")
        elif not re.search("[A-Z]", pass1):
            messages.error(request,"Password should contain atleast one Uppercase letter")
        elif not re.search("[0-9]", pass1):
            messages.error(request,"Password should contain atleast one Number")
        elif not re.search("[_@!#%$]", pass1):
            messages.error(request,"Password should contain atleast one Special character")
        elif mobno.isnumeric() == False or len(mobno) != 10 :
            messages.error(request,"Enter a valid Mobile number")
        else :    
            myuser=User.objects.create_user(username=nusername, password=pass1, email=nemail)
            myuser.first_name = fname
            myuser.last_name = lname
            myuser.save()
            newuserprofile=Input_Image(user = myuser)
            newuserprofile.save()

            messages.success(request, "Your account has been successfully created!")

        return redirect('/signin')   # to redirect the user to the signin page once the successful registration messages is displayed
    
    return render(request, "ourapp/signup.html")

def signin(request):

    if request.method == 'POST':
        username = request.POST['username']
        pass1 = request.POST['pass1']
        
        user = authenticate(username = username, password = pass1)

        if user is not None:
            login(request, user)
            fname = user.first_name

            return redirect('/upload')

        else:
            messages.error(request, "Bad Credentials")
            return render(request, "ourapp/signin.html")

    return render( request, "ourapp/signin.html")

def signout(request):
    logout(request)
    messages.success(request, "Logged out successfully!")
    return redirect('/home')


def upload(request):
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)

        if form.is_valid():
            # input_image = Input_Image.objects.filter(user = request.user)
            # input_image = form.save(commit=False)
            # input_image.user = request.user
            # input_image.save()

            # form.save(commit_user=request.user)
            # obj = form.instance

            user = request.user
            obj = Input_Image.objects.get_or_create(user=user)
            obj = obj[0]
            obj.image = form.cleaned_data['image']
            obj.save()
            return render(request, 'ourapp/upload.html', {'obj':obj})
        
    else:
        form = ImageForm()
        # img = Input_Image.objects.filter(user = request.user)
        img = Input_Image.objects.all()
    return render(request, 'ourapp/upload.html',{"img":img, "form":form})

def detection(request):
    print("In here detection function")
    CONFIDENCE = 0.5
    SCORE_THRESHOLD = 0.5
    IOU_THRESHOLD = 0.5

    ruser = request.user
    print(ruser)
    input_image = Input_Image.objects.get(user = ruser)
    processed_image = Processed_Image(user = ruser)

    # if request.method == 'POST':
    # input_image.
    # the neural network configuration
    config_path = "ourapp/yolov3.cfg.txt"

    # the YOLO net weights file
    weights_path = "ourapp/yolov3.weights"


    # loading all the class labels (objects)
    labels = open("ourapp/coco.names.txt").read().strip().split("\n")

    # generating colors for each object for later plotting
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")


    net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
    image_path = input_image.image.path
    # imagefile = input_image.image
    image = cv2.imread(image_path)
    file_name = os.path.basename(image_path)
    filename, ext = file_name.split(".")
    
    h, w = image.shape[:2]

    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    try:
        ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    except IndexError:
        # in case getUnconnectedOutLayers() returns 1D array when CUDA isn't available
        ln = [ln[i - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start

    font_scale = 1
    thickness = 3
    boxes, confidences, class_ids, ext_coordinates = [], [], [], []

    # loop over each of the layer outputs
    for output in layer_outputs:
        
        # loop over each of the object detections
        for detection in output:
            
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE and confidence >= 0.9:

                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    #print(detection.shape)
    print("Coordinates:")

    if(len(boxes) >= 1):
        pre_x = boxes[0][0]
        pre_y = boxes[0][1]

        ext_coordinates.append([boxes[0][0], boxes[0][1], boxes[0][2], boxes[0][3]])

    # loop over the indexes we are keeping
    for i in range(len(boxes)):
        # extract the bounding box coordinates
        x, y = boxes[i][0], boxes[i][1]
        w, h = boxes[i][2], boxes[i][3]

        print(x, y, w, h)

        if(x <= pre_x - 50 or x >= pre_x + 50):
            if (y <= pre_y - 20 or y >= pre_y + 20):
                ext_coordinates.append([x, y, w, h])
                pre_x = x
                pre_y = y
        

        # draw a bounding box rectangle and label on the image
        color = [int(c) for c in colors[class_ids[i]]]
        cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
        text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"
        

        # calculate text width & height to draw the transparent boxes as background of the text
        (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
        text_offset_x = x
        text_offset_y = y - 5
        box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
        overlay = image.copy()
        cv2.rectangle(overlay, box_coords[0], box_coords[1], color, thickness=cv2.FILLED)
        

        # add opacity (transparency to the box)
        image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
        

        # now put the text (label: confidence %)
        cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,font_scale, (0, 0, 0), thickness)
        # cv2.imwrite(filename + "_yolo3." + ext, image)
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)

        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            x_pre = boxes[0][0]
            y_pre = boxes[0][1]
            

            for i in idxs.flatten():
                # extract the bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]


                # draw a bounding box rectangle and label on the image
                color = [int(c) for c in colors[class_ids[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
                text = f"{labels[class_ids[i]]}: {confidences[i]:.2f}"


                # calculate text width & height to draw the transparent boxes as background of the text
                (text_width, text_height) = \
                cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                text_offset_x = x
                text_offset_y = y - 5
                box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
                overlay = image.copy()
                cv2.rectangle(overlay, box_coords[0], box_coords[1], color, thickness=cv2.FILLED)
                

                # add opacity (transparency to the box)
                image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)

                # now put the text (label: confidence %)
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,font_scale, (0, 0, 0), thickness)

    
    if input_image.detect == False:

        for k in range(len(ext_coordinates)):
            print(ext_coordinates[k][0], ext_coordinates[k][1],
                ext_coordinates[k][2], ext_coordinates[k][3])
            input = cv2.imread(image_path)
            output = remove(crop_image(
                input, ext_coordinates[k][0], ext_coordinates[k][1], ext_coordinates[k][2], ext_coordinates[k][3]))
            
            ext_path = filename + str(k) + "_Output.png"

            output_pil = Image.fromarray(output)
            output_pil.save(ext_path, format='PNG')
            processed_image.ext_img.save("extracted_image.jpg", ImageFile(open(ext_path, 'rb')))
            # ext_img = Processed
        
            # cv2.imwrite(filename + str(k) + "_Output.png", output)
 
            # processed_image.save()
            # return render(request, ourapp)
            print(k)
            processed_image.save()

        input_image.save()
        processed_image = Processed_Image.objects.filter(user = request.user)
        return render(request, 'ourapp/extraction.html', {'processed_image':processed_image})


    if input_image.detect == True:

        new_path = filename + "_yolo3." + ext 

        input_image.detect = False
        image_pil = Image.fromarray(image) #converting numpy array into image using the Image function of thr Pillow Library
        image_pil.save(new_path, format='PNG')
        input_image.det_img.save("detected_image.jpg", ImageFile(open(new_path, 'rb')))
        input_image.save()

        print("In here after save")
        return render(request, 'ourapp/display.html', {'input_image':input_image})



def crop_image(img, x, y, w, h):
    return img[y-10:y+h+10, x-10:x+w+10]

