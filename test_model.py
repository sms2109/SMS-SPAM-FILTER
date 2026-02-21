import pickle

model = pickle.load(open("model.pkl", "rb"))

print("Model Loaded Successfully")
print(model)

# Try checking if fitted
try:
    print(model.classes_)
    print("Model is fitted ✅")
except:
    print("Model is NOT fitted ❌")