# Importing Necessary Libraries
import tkinter as tk
from tkinter import filedialog
from tkinter import *
from PIL import Image,ImageTk
import numpy as np
import tensorflow
from tensorflow.keras.models import load_model

# loading the model 
model = load_model("Final_Resnet50_Best_model.keras")
bgcolor = '#990011'
fgcolor = '#FCF6F5'

# Initializing the GUI
top=tk.Tk()
top.geometry('800x600')
top.title('Emotion Detection')
top.configure(background= bgcolor)

label1=Label(top,background=fgcolor,font=('arial',15,"bold"))
sign_image=Label(top)

def Detect(file_path):
    global label_packed
    
    image=Image.open(file_path)

    image=image.resize((224,224))
    image=np.expand_dims(image,axis=0)
    image=np.array(image, dtype=np.float32)
    
    image=np.delete(image,0,1)
    image=np.resize(image,(224,224,3))
    

    
    image=np.array([image], dtype=np.float32)/255.0
    
    prediction = model.pred(image)
    emotion_dict = {0: 'angry',
                    1: 'disgust',
                    2: 'fear',
                    3: 'happy',
                    4: 'neutral',
                    5: 'sad',
                    6: 'surprise'}
    pred=int(np.round(prediction[0][0]))
    print(prediction, pred)
    emotion = emotion_dict[pred]
    output_text = f"He/she is feeling {emotion}"
    label1.configure(foreground=bgcolor,text=output_text)

# Defining Show_detect button function
def show_Detect_button(file_path):
    Detect_b=Button(top,text="Detect Image",command=lambda: Detect(file_path),padx=10,pady=5)
    Detect_b.configure(background=fgcolor,foreground=bgcolor,font=('arial',10,'bold'))
    Detect_b.place(relx=0.79,rely=0.46) 

# Definig Upload Image Function
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)

        sign_image.configure(image=im)
        sign_image.image=im
        label1.configure(text='')
        show_Detect_button(file_path)
    except:
        pass

upload=Button(top,text="Upload an Image",command=upload_image,padx=10,pady=5)
upload.configure(background=fgcolor,foreground=bgcolor,font=('arial',10,'bold'))
upload.pack(side='bottom',pady=50)
sign_image.pack(side='bottom',expand=True)
label1.pack(side="bottom",expand=True)
heading=Label(top,text="Emotion Detection",pady=20,font=('arial',20,"bold"))
heading.configure(background=fgcolor,foreground=bgcolor)
heading.pack()
top.mainloop()