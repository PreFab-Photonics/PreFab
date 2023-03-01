"""Example prediction of a simple device using a PreFab model.
"""

# %% imports
import sys
import matplotlib.pyplot as plt
sys.path.append('../')
import prefab as pf  # pylint: disable=E0401 C0413  # noqa: E402

# %% set key parameters
MODEL_TYPE = 'p'
MODEL_NAME = 'ANT_NanoSOI'
MODEL_VERSION = 'v2'

predictor = pf.Predictor(model_type=MODEL_TYPE, model_name=MODEL_NAME,
                         model_version=MODEL_VERSION)

# %% load shape (device) and show
DEVICE_LENGTH = 200
device = pf.load_device_img(path='devices/rectangle_128x128_256x256.png',
                            device_length=DEVICE_LENGTH)
device = pf.load_device_gds(path='devices/devices.gds', cell_name='cross')

plt.imshow(device)
plt.title('\nNominal')
plt.ylabel('Distance (nm)')
plt.xlabel('Distance (nm)')
plt.show()

# %% make prediction
STEP_LENGTH = 8
prediction = predictor.predict(device=device, step_length=STEP_LENGTH,
                               binary=False)

# %% plot prediction
plt.imshow(prediction)
plt.title('Prediction')
plt.show()

# %%
