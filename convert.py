import tensorflow as tf

model = tf.keras.models.load_model("lung_xray_model.keras")
model.save("lung_xray_model.h5", save_format="h5")

print("Model converted successfully!")