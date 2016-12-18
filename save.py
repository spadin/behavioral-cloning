import json

def save(model, filename="model.json"):
    with open(filename, "w") as f:
        json.dump(model.to_json(), f)

    model.save_weights(filename.replace("json","h5"))

