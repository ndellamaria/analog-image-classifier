"""Run once: python convert_to_onnx.py  (requires tf2onnx — pip install tf2onnx)"""
import tensorflow as tf
import subprocess
import shutil

model = tf.keras.models.load_model('model.keras')

tmp = 'saved_model_tmp'
model.export(tmp)

subprocess.run(
    ['python', '-m', 'tf2onnx.convert', '--saved-model', tmp, '--output', 'model.onnx'],
    check=True
)

shutil.rmtree(tmp)
print("Done — model.onnx written")
