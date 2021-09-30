import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv

train = pd.read_csv('data/train.csv')
image = train['image'].iloc[1]
im  = cv.imread(f"data/train_images/{image}")
plt.imshow(im)
plt.show()

print(train['title'].iloc[1])
print(train['label_group'])