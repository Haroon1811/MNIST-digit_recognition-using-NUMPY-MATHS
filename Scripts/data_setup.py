 
import os
from pathlib import Path
import zipfile
  
data_path = Path(path)
image_path = data_path / "digit_recognizer"

if image_path.is_dir():
  print(f"{image_path} directory already exists .... skipp creating one")
else:
  print(f"{image_path} does not exist, creating one...")
  image_path.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(data_path / "digit-recognizer.zip", "r") as f:
    print("Unzipping the file...")
    f.extractall(image_path)
    print("Done")

# Read csv data from the unzipped folder
data = pd.read_csv("Data/digit-recognizer/digit_recognizer/train.csv")
# convert the DataFrame into numpy array
data = np.array(data)
m, n = data.shape            # number of rows,m and columns,n 
# Shuffle before spliting 
np.random.shuffle(data)

data_val = data[0:1000].T   # transpose taken because the matrix multiplication requires the shape match 
Y_val = data_val[0]         # because the zeroth column is of label(image drawn by the user)
X_val = data_val[1:n]
"""
Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel,
with higher numbers meaning darker. This pixel-value is an integer between 0 and 255, inclusive.
"""
X_val = X_val/255           # each pixel has value between 0-255

data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train/255
