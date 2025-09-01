# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

Sturdy Octo Disco is a fun project that adds sunglasses to photos using image processing.

Welcome to Sturdy Octo Disco, a fun and creative project designed to overlay sunglasses on individual passport photos! This repository demonstrates how to use image processing techniques to create a playful transformation, making ordinary photos look extraordinary. Whether you're a beginner exploring computer vision or just looking for a quirky project to try, this is for you!

## Features:
- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:
- Python
- OpenCV for image processing
- Numpy for array manipulations

## How to Use:
1. Clone this repository.
2. Add your passport-sized photo to the `images` folder.
3. Run the script to see your "cool" transformation!

## Applications:
- Learning basic image processing techniques.
- Adding flair to your photos for fun.
- Practicing computer vision workflows.

Feel free to fork, contribute, or customize this project for your creative needs!

## Program :

```python
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('MYPHOTO.jpg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```

<img width="468" height="577" alt="image" src="https://github.com/user-attachments/assets/19ca278e-feba-48f6-b1d5-d013185b6d0e" />

```python
faceImage.shape
```

<img width="123" height="36" alt="image" src="https://github.com/user-attachments/assets/12dac66f-6b84-46f9-91aa-17f1e93453b8" />

```python
# Load the Sunglass image with Alpha channel
# (http://pluspng.com/sunglass-png-1104.html)
glassPNG = cv2.imread('sunglass.png',-1)
plt.imshow(glassPNG[:,:,::-1]);plt.title("glassPNG")
```

<img width="719" height="387" alt="image" src="https://github.com/user-attachments/assets/2ac9bc35-c1b7-40a4-9eb9-8501a6c892ca" />

```python
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG,(135,65))
print("image Dimension ={}".format(glassPNG.shape))
```
<img width="275" height="30" alt="image" src="https://github.com/user-attachments/assets/6fcd1852-509e-49d3-a7d3-4b6a0405d6f2" />

```python
# Separate the Color and alpha channels
glassBGR = glassPNG[:,:,0:3]
glassMask1 = glassPNG[:,:,3]

# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="1379" height="375" alt="image" src="https://github.com/user-attachments/assets/2419de02-4ce3-4ffd-9508-135064b95894" />

```python
# Make a copy
#faceWithGlassesNaive = resized_faceImage.copy()
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[170:235, 135:270] = glassBGR

plt.imshow(faceWithGlassesNaive[...,::-1])
```

<img width="450" height="557" alt="image" src="https://github.com/user-attachments/assets/7c9bf8b8-0f96-4b56-be0f-d312e1ac3fed" />

```python
# Make the dimensions of the mask same as the input image.
# Since Face Image is a 3-channel image, we create a 3 channel image for the mask
glassMask = cv2.merge((glassMask1,glassMask1,glassMask1))

# Make the values [0,1] since we are using arithmetic operations
glassMask = np.uint8(glassMask/255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[170:235, 135:270]   # shape (40,120,3)

# Resize glass image and mask to match ROI
glassBGR_resized = cv2.resize(glassBGR, (eyeROI.shape[1], eyeROI.shape[0]))
glassMask_resized = cv2.resize(glassMask, (eyeROI.shape[1], eyeROI.shape[0]))

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask_resized))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR_resized, glassMask_resized)

# Combine the Sunglass in the Eye Region
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=[20,20])
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
```

<img width="1368" height="283" alt="image" src="https://github.com/user-attachments/assets/3b73c92a-83ec-427a-ad6a-63039d325161" />

```python
# Replace the eye ROI with the output from the previous section
faceWithGlassesArithmetic[170:235, 135:270]=eyeRoiFinal

# Display the final result
plt.figure(figsize=[20,20]);
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]); plt.title("Original Image");
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses");
```

<img width="1230" height="741" alt="image" src="https://github.com/user-attachments/assets/9efa4ea1-240f-44a0-99af-042893e201b6" />
