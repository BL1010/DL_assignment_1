import numpy as np

w = np.load("src/best_model.npy", allow_pickle=True)

print("num weights:", len(w))

for i in range(len(w)):
    print(i, np.array(w[i]).shape)