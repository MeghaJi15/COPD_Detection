import tensorflow as tf

# load keras model
model = tf.keras.models.load_model("lung_xray_model.keras", compile=False)

# rebuild input layer (fix compatibility)
model.build((None,224,224,3))

# save compatible model
model.save("lung_xray_model_fixed.h5")

print("Model fixed and saved!")