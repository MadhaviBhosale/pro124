# To Capture Frame
import cv2

# To process image array
import numpy as np

#Importing tensorflow
import tensorflow as tf


# import the tensorflow modules and load the model
model = tf.keras.models.load_model('keras_model.h5')



# Attaching Cam indexed as 0, with the application software
camera = cv2.VideoCapture(0)

# Infinite loop
while True:
    status , frame = camera.read()
    frame = cv2.flip(frame , 1)
    image = cv2.resize(frame,(244,244))
    test_image = np.array(image,dtype=np.float32)
    test_image = np.expand_dims(test_image,axis=0)
    normalise = test_image/255.0
    prediction = model.predict(frame)
    print('Prediction : ',prediction)
    cv2.imshow('feed' , frame)
    code = cv2.waitKey(1)
    if code == 32:
        break
    

	# Reading / Requesting a Frame from the Camera 
	
 

		
		#resize the frame
		
		# expand the dimensions
		
		# normalize it before feeding to the model
		
		# get predictions from the model
		
		
		
		
		

# release the camera from the application software
camera.release()

# close the open window
cv2.destroyAllWindows()