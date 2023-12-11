import streamlit as st
import matplotlib.pyplot as plt
import io
import numpy as np
import torch
import torchaudio
# import matplotlib.pylab as plt
import pyaudio
import omegaconf
import pygame
import librosa
import soundfile as sf
import tensorflow as tf
import tensorflow_hub as hub
import csv
import librosa
# from utils_vad import calculate_noise_level, int2float, calculate_loudness



st.title('Fall detection')


# 일 소음 정도를 측정하기 위한 코드

# creating fall detection model
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, MultiHeadAttention, LayerNormalization, Dropout, Flatten
from tensorflow.keras.models import Model

# Hyperparameters based on the table you provided
input_shape = (88, 1600) # (N+1 sequences, D dimension of each sequence)
batch_size = 20
epochs = 10
learning_rate = 0.00001
encoder_layers = 12
num_heads = 12
d_model = 512
ff_dim = 1024
dropout_rate = 0.1

# Input Layer
input_tensor = Input(shape=input_shape)

# Assuming `input_tensor` has shape (batch_size, 88, 1600)
# Create a CLS token with the same type as the input_tensor (e.g., float32)
cls_token = tf.constant(1, shape=(1, 1600), dtype=input_tensor.dtype)
cls_token = tf.expand_dims(cls_token, 0)  # Expand dims to (1, 1, 1600)

# Tile the CLS token to match the batch size of input_tensor
cls_token = tf.tile(cls_token, [tf.shape(input_tensor)[0], 1, 1])  

# Concatenate along the sequence axis
input_with_cls = tf.keras.layers.Concatenate(axis=1)([cls_token, input_tensor])



# Linear Projection
linear_projection = Dense(d_model)(input_with_cls)

# Positional Encoding - Not explicitly shown here, but would be added to linear_projection
# ...

# Transformer Encoder Layers
def transformer_encoder_block(x):
    attention_output = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)(x, x)
    attention_output = Dropout(dropout_rate)(attention_output)
    attention_output = LayerNormalization(epsilon=1e-6)(attention_output + x)

    ffn_output = Dense(ff_dim, activation="relu")(attention_output)
    ffn_output = Dense(d_model)(ffn_output)
    ffn_output = Dropout(dropout_rate)(ffn_output)
    return LayerNormalization(epsilon=1e-6)(ffn_output + attention_output)

x = linear_projection
for _ in range(encoder_layers):
    x = transformer_encoder_block(x)

# Flatten for MLP
x = Flatten()(x)

# MLP Block
mlp_block = Dense(265, activation='relu', name='mlp_dense_1')(x)
mlp_block = Dense(64, activation='relu', name='mlp_dense_2')(mlp_block)
mlp_block = Dense(10, activation='relu', name='mlp_dense_3')(mlp_block)


# Output Layer
output_tensor = Dense(2, activation='softmax', name='output_layer')(mlp_block) # 2 classes: fall, no fall
# output_tensor = Dense(2, activation='softmax')(mlp_block) # 2 classes: fall, no fall

# Create the model
model = Model(inputs=input_tensor, outputs=output_tensor)

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Model summary
model.summary()

model.save('fall_detection_model')





# model save
vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad',
                              model='silero_vad',
                              force_reload=True)

yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

fall_detection_model = tf.keras.models.load_model('fall_detection_model')




# define utils
(get_speech_timestamps,
 save_audio,
 read_audio,
 VADIterator,
 collect_chunks) = utils



# helper method
def validate(model,
             inputs: torch.Tensor):
    with torch.no_grad():
        outs = model(inputs)
    return outs

# Provided by Alexander Veysov
def int2float(sound):
    abs_max = np.abs(sound).max()
    sound = sound.astype('float32')
    if abs_max > 0:
        sound *= 1/32768
    sound = sound.squeeze()  # depends on the use case
    return sound

# Function to play an audio alert
def play_audio_alert(audio_file):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file)
    pygame.mixer.music.play()

# Calculate loudness of audio chunk
def calculate_loudness(audio_chunk):
    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
    audio_data = audio_data.astype(np.float64)  # Convert to float64 to prevent overflow
    if audio_data.size == 0:
        return 0  # Return 0 or an appropriate value for empty data
    return np.sqrt(np.mean(audio_data**2))


# Estimate pitch of audio chunk
def estimate_pitch(y, sr, fmin=150.0, fmax=4000.0):
    pitches, magnitudes = librosa.core.piptrack(y=y, sr=sr, fmin=fmin, fmax=fmax)
    index = magnitudes.argmax()
    pitch = pitches[index]
    return pitch


def make_prediction(input_data, threshold=0.5):
    """
    Makes a prediction based on the input data.

    :param input_data: A NumPy array or a list of lists with the shape matching the model's input shape.
    :return: The prediction result.
    """
    # Ensure the input data is a NumPy array
    if not isinstance(input_data, np.ndarray):
        input_data = np.array(input_data)

    # Reshape or process the input_data as necessary for your model
    # ...

    # Make a prediction
    prediction = fall_detection_model.predict(input_data)

    is_fall = prediction[0][1] >= threshold  # Considering 'fall' is the second class
    return is_fall




# pre-recording background noise
def calculate_noise_level(audio_data):
    return np.mean([np.std(chunk) for chunk in audio_data])


# Set the parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
SAMPLE_RATE = 16000 
CHUNK = int(SAMPLE_RATE / 10)
frames_to_record = 160 # speech/noise
# pre_record_chunks = int(st.slider('주변 소음을 몇 초동안 수집할지 설정하세요', 0, 200, 100) * SAMPLE_RATE // CHUNK)  # Number of chunks for measuring noise
pre_record_chunks = 100  # Number of chunks for measuring noise
threshold = 0.5  # Threshold for speech detection
audio = pyaudio.PyAudio()


# Initialize audio stream and pre-record for noise measurement
stream = pyaudio.PyAudio().open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK, input_device_index=0)

# stream = pyaudio.PyAudio().open(format=FORMAT, channels=CHANNELS, rate=SAMPLE_RATE, input=True, frames_per_buffer=CHUNK)

# 슬라이드 바를 통해 노이즈 레벨을 측정하는 함수
def measure_noise_level(duration_sec):
    st.write(f"노이즈 레벨을 {duration_sec}초 동안 측정 중입니다...")
    pre_record_chunks = int(duration_sec * SAMPLE_RATE // CHUNK)
    pre_record_data = [np.frombuffer(stream.read(CHUNK), np.int16) for _ in range(pre_record_chunks)]
    noise_level = calculate_noise_level(pre_record_data)
    st.write(f"Noise level: {noise_level}")
    st.write("측정이 완료되었습니다.")

# 슬라이드 바를 통해 노이즈 레벨을 측정할지 여부를 선택하는 부분
duration_sec = st.slider('노이즈 레벨을 측정할 시간(초)을 선택하세요', 1, 10, 5)
if st.button('노이즈 레벨 측정 시작'):
    measure_noise_level(duration_sec)


# # Use st.button to trigger noise level measurement
# if st.button('노이즈 레벨 측정 시작'):
#     pre_record_data = [np.frombuffer(stream.read(CHUNK), np.int16) for _ in range(pre_record_chunks)]
#     noise_level = calculate_noise_level(pre_record_data)
#     st.write(f"Noise level: {noise_level}")


# pre_record_data = [np.frombuffer(stream.read(CHUNK), np.int16) for _ in range(pre_record_chunks)]

# noise_level = calculate_noise_level(pre_record_data)
# st.write(f"Noise level: {noise_level}")

data = []
voiced_confidences = []


# ==================
# streamlit code




if st.button('noise감지 시작'):
    # Start recording
    st.write("Started Recording")

    for i in range(0, frames_to_record):
        audio_chunk = stream.read(CHUNK)
        data.append(audio_chunk)

        audio_int16 = np.frombuffer(audio_chunk, np.int16)
        audio_float32 = int2float(audio_int16)

        # Convert the audio to a suitable format for YAMNet
        waveform = np.frombuffer(audio_chunk, dtype=np.int16) / 32768.0
        waveform = waveform.astype(np.float32)

        # Predict with YAMNet
        # ...

        # get the confidences and add them to the list to plot them later
        confidence = vad_model(torch.from_numpy(audio_float32), SAMPLE_RATE).item()
        # confidence = 0.5  # Replace this with the actual confidence calculation
        voiced_confidences.append(confidence)

        # Classify and print the result
        if confidence > threshold:
            st.write(f"Chunk {i+1}: Speech")
        else:
            st.write(f"Chunk {i+1}: Noise")

    st.write("Stopped the recording")

    # # Plot the confidences for the speech
    # st.pyplot(plt.plot(voiced_confidences))

    fig, ax = plt.subplots()
    ax.plot(voiced_confidences)
    ax.set_xlabel('X-axis Label')  # Add your X-axis label
    ax.set_ylabel('Y-axis Label')  # Add your Y-axis label
    ax.set_title('Title')  # Add your title

    # Display the figure using st.pyplot
    st.pyplot(fig)


