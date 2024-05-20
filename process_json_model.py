import json
from tensorflow.keras.models import model_from_json
from tensorflow.keras.initializers import Orthogonal

# Load model architecture
with open('pos_model_architecture.json', 'r') as f:
    model_json = json.load(f)

# Remove unsupported arguments from the architecture
for layer in model_json['config']['layers']:
    if 'config' in layer and 'time_major' in layer['config']:
        del layer['config']['time_major']

# Rebuild the model
model = model_from_json(json.dumps(model_json), custom_objects={'Orthogonal': Orthogonal})

# Load the weights
model.load_weights('pos_model.h5')
