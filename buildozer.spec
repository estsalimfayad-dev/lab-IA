[app]
title = Detector de caidas
package.name = detectorcaidas
package.domain = co.edu.umng
source.dir = .
source.include_exts = py,kv,npz,json
version = 1.0
# numpy + kivy + plyer + pyjnius: sin TensorFlow (el LSTM corre en NumPy, ver nucleo.py)
requirements = python3,kivy==2.3.0,numpy,plyer,pyjnius,android
orientation = portrait
fullscreen = 0
android.permissions = VIBRATE,WAKE_LOCK,READ_EXTERNAL_STORAGE,HIGH_SAMPLING_RATE_SENSORS
android.api = 34
android.minapi = 24
android.archs = arm64-v8a
android.accept_sdk_license = True

[buildozer]
log_level = 2
warn_on_root = 1
