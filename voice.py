# from collections import deque
# import os
# from dotenv import load_dotenv

# import numpy as np
# import pvporcupine
# import pvcobra
# import whisper
# from pvrecorder import PvRecorder
# import torch

# load_dotenv()

# porcupine = pvporcupine.create(
#     # access_key=os.environ.get("ACCESS_KEY"),
#     keyword_paths=[os.environ.get("WAKE_WORD_MODEL_PATH")],
# )

# cobra = pvcobra.create(
#     # access_key=os.environ.get("ACCESS_KEY"),
# )

# recoder = PvRecorder(device_index=-1, frame_length=512)

# # frame length = 512
# # samples per frame = 16,000
# # 1 sec = 16,000 / 512


# class Transcriber:
#     def __init__(self, model) -> None:
#         print("loading model")
#         # TODO: put model on GPU
#         self.model = whisper.load_model(model)
#         print("loading model finished")
#         self.prompts = os.environ.get("WHISPER_INITIAL_PROMPT", "")
#         print(f"Using prompts: {self.prompts}")

#     def transcribe(self, frames):
#         transcribe_start = time.time()
#         samples = np.array(frames, np.int16).flatten().astype(np.float32) / 32768.0

#         # audio = whisper.pad_or_trim(samples)
#         # print(f"{transcribe_start} transcribing {len(frames)} frames.")
#         # # audio = whisper.pad_or_trim(frames)

#         # # make log-Mel spectrogram and move to the same device as the model
#         # mel = whisper.log_mel_spectrogram(audio).to(self.model.device)

#         # # decode the audio
#         # options = whisper.DecodingOptions(fp16=False, language="english")
#         # result = whisper.decode(self.model, mel, options)

#         result = self.model.transcribe(
#             audio=samples,
#             language="en",
#             fp16=False,
#             initial_prompt=self.prompts,
#         )

#         # print the recognized text
#         transcribe_end = time.time()
#         # print(
#         #     f"{transcribe_end} - {transcribe_end - transcribe_start}sec: {result.get('text')}",
#         #     flush=True,
#         # )
#         return result.get("text", "speech not detected")


# transcriber = Transcriber(os.environ.get("WHISPER_MODEL"))

# sample_rate = 16000
# frame_size = 512
# vad_mean_probability_sensitivity = float(os.environ.get("VAD_SENSITIVITY"))

# try:
#     recoder.start()

#     max_window_in_secs = 3
#     window_size = sample_rate * max_window_in_secs
#     samples = deque(maxlen=(window_size * 6))
#     vad_samples = deque(maxlen=25)
#     is_recording = False

#     while True:
#         data = recoder.read()
#         vad_prob = cobra.process(data)
#         vad_samples.append(vad_prob)
#         # print(f"{vad_prob} - {np.mean(vad_samples)} - {len(vad_samples)}")
#         if porcupine.process(data) >= 0:
#             print(f"Detected wakeword")
#             is_recording = True
#             samples.clear()

#         if is_recording:
#             if (
#                 len(samples) < window_size
#                 or np.mean(vad_samples) >= vad_mean_probability_sensitivity
#             ):
#                 samples.extend(data)
#                 print(f"listening - samples: {len(samples)}")
#             else:
#                 print("is_recording: False")
#                 print(transcriber.transcribe(samples))
#                 is_recording = False
# except KeyboardInterrupt:
#     recoder.stop()
# finally:
#     porcupine.delete()
#     recoder.delete()
#     cobra.delete()

# import os
# import sounddevice as sd
# import soundfile as sf
# from openai import OpenAI
# from decouple import config
# from os import getenv
# from dotenv import load_dotenv

# load_dotenv()

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to record audio
# def record_audio(file_path, duration=5, fs=48000, device_index=None):
#     print("Recording...")
#     audio_data = sd.rec(int(duration * fs), samplerate=fs, channels=1, device=device_index)
#     sd.wait()  # Wait until recording is finished
#     sf.write(file_path, audio_data, fs)
#     print(f"Recording saved successfully to {file_path}")

# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(file_path):
#     with open(file_path, "rb") as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from OpenAI GPT-4o
# def get_openai_llm_response(transcribed_text):
#     prompt = f"You are a helpful assistant named Rica. Answer in one sentence with humor: {transcribed_text}"
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message['content']

# # Function to generate speech using OpenAI TTS model
# def generate_speech(text_input, output_path):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
#     with open(output_path, "wb") as file:
#         for chunk in response.iter_bytes():
#             file.write(chunk)

# # Main script execution
# if __name__ == "__main__":
#     audio_file_path = 'audio/question.wav'
#     tts_audio_file_path = 'audio/answer.mp3'

#     # Record audio from the user
#     duration = int(input("Enter recording duration in seconds: "))
#     record_audio(audio_file_path, duration)

#     # Transcribe audio to text
#     transcription_text = transcribe_audio(audio_file_path)
#     print(f"You said: {transcription_text}")

#     # Get response from LLM
#     llm_response = get_openai_llm_response(transcription_text)
#     print(f"AI says: {llm_response}")

#     # Generate speech from LLM response
#     generate_speech(llm_response, tts_audio_file_path)
#     print(f"Response has been converted to speech and saved to {tts_audio_file_path}")

#     # Optionally play the audio (if desired, may require additional libraries)
#     # os.system(f"start {tts_audio_file_path}")  # Windows
#     # os.system(f"afplay {tts_audio_file_path}")  # macOS


# import os
# import soundfile as sf
# import speech_recognition as sr
# from openai import OpenAI
# from decouple import config

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(audio_data):
#     # Save the audio data to a temporary file
#     with open('temp_audio.wav', 'wb') as temp_file:
#         sf.write(temp_file, audio_data.get_raw_data(), 44100)

#     with open('temp_audio.wav', 'rb') as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from OpenAI GPT-4o
# def get_openai_llm_response(transcribed_text):
#     prompt = f"You are a helpful assistant named Rica. Answer in one sentence with humor: {transcribed_text}"
#     response = client.chat.completions.create(
#         model="gpt-4o",
#         messages=[{"role": "user", "content": prompt}]
#     )
#     return response.choices[0].message['content']

# # Function to generate speech using OpenAI TTS model
# def generate_speech(text_input, output_path):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
#     with open(output_path, "wb") as file:
#         for chunk in response.iter_bytes():
#             file.write(chunk)

# # Main script execution
# if __name__ == "__main__":
#     tts_audio_file_path = 'audio/answer.mp3'
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     print("Press 'Enter' to start recording...")
#     input()  # Wait for the Enter key to start recording

#     with microphone as source:
#         print("Recording... Press 'Enter' again to stop.")
#         while True:
#             try:
#                 # Listen for audio and recognize it
#                 audio_data = recognizer.listen(source, timeout=5)
#                 transcription = recognizer.recognize_whisper(audio_data)
#                 print(f"You said: {transcription}")

#                 # Optional: Transcribe using OpenAI Whisper
#                 # transcription_text = transcribe_audio(audio_data)
#                 # print(f"You said: {transcription_text}")

#             except sr.WaitTimeoutError:
#                 print("Listening timeout. Press 'Enter' to stop.")
#                 break
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from the service; {e}")
#             except KeyboardInterrupt:
#                 print("Recording stopped.")
#                 break

#     # Final transcription and response generation
#     if 'transcription' in locals():
#         llm_response = get_openai_llm_response(transcription)
#         print(f"AI says: {llm_response}")
#         generate_speech(llm_response, tts_audio_file_path)
#         print(f"Response has been converted to speech and saved to {tts_audio_file_path}")


# import os
# import soundfile as sf
# import speech_recognition as sr
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from decouple import config
# from openai import OpenAI

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()


# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(audio_data):
#     with open('temp_audio.wav', 'wb') as temp_file:
#         sf.write(temp_file, audio_data.get_raw_data(), 44100)

#     with open('temp_audio.wav', 'rb') as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from the LLM using Langchain
# def get_openai_llm_response(transcribed_text):
#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rica who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE IN ENGLISH LANGUAGE ONLY."),
#             ("human", "{input}"),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4o")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input=transcribed_text)
    
#     return response.content

# # Function to generate speech using OpenAI TTS model
# def generate_speech(text_input, output_path):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
#     with open(output_path, "wb") as file:
#         for chunk in response.iter_bytes():
#             file.write(chunk)

# # Main script execution
# if __name__ == "__main__":
#     tts_audio_file_path = 'audio/answer.mp3'
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     print("Press 'Enter' to start recording...")
#     input()  # Wait for the Enter key to start recording

#     transcription = None  # Initialize transcription variable

#     with microphone as source:
#         print("Recording... Press 'Enter' again to stop.")
#         while True:
#             try:
#                 # Listen for audio and recognize it
#                 audio_data = recognizer.listen(source, timeout=5)
#                 transcription = recognizer.recognize_whisper(audio_data)
#                 print(f"You said: {transcription}")

#                 # Use the transcribed text for further processing
#                 llm_response = get_openai_llm_response(transcription)
#                 print(f"AI says: {llm_response}")

#                 # Generate speech from LLM response
#                 generate_speech(llm_response, tts_audio_file_path)
#                 print(f"Response has been converted to speech and saved to {tts_audio_file_path}")

#             except sr.WaitTimeoutError:
#                 print("Listening timeout. Press 'Enter' to stop.")
#                 break
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from the service; {e}")
#             except KeyboardInterrupt:
#                 print("Recording stopped.")
#                 break

#     # Final check for transcription
#     if transcription:
#         print("Final transcription was successful.")
#     else:
#         print("No transcription was received.")



# import os
# import soundfile as sf
# import speech_recognition as sr
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from decouple import config
# from openai import OpenAI
# import pydub
# import io
# from pydub.playback import play  # Import the play function

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(audio_data):
#     with open('temp_audio.wav', 'wb') as temp_file:
#         sf.write(temp_file, audio_data.get_raw_data(), 44100)

#     with open('temp_audio.wav', 'rb') as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from the LLM using Langchain
# def get_openai_llm_response(transcribed_text):
#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rica who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE IN ENGLISH LANGUAGE ONLY."),
#             ("human", "{input}"),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4o")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input=transcribed_text)
    
#     return response.content

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
    
#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = pydub.AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Main script execution
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     print("Press 'Enter' to start recording...")
#     input()  # Wait for the Enter key to start recording

#     transcription = None  # Initialize transcription variable

#     with microphone as source:
#         print("Recording... Press 'Enter' again to stop.")
#         while True:
#             try:
#                 # Listen for audio and recognize it
#                 audio_data = recognizer.listen(source, timeout=5)
#                 transcription = recognizer.recognize_whisper(audio_data)
#                 print(f"You said: {transcription}")

#                 # Use the transcribed text for further processing
#                 llm_response = get_openai_llm_response(transcription)
#                 print(f"AI says: {llm_response}")

#                 # Generate and play speech from LLM response
#                 generate_and_play_speech(llm_response)

#             except sr.WaitTimeoutError:
#                 print("Listening timeout. Press 'Enter' to stop.")
#                 break
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from the service; {e}")
#             except KeyboardInterrupt:
#                 print("Recording stopped.")
#                 break

#     # Final check for transcription
#     if transcription:
#         print("Final transcription was successful.")
#     else:
#         print("No transcription was received.")


# #Final
# import os
# import soundfile as sf
# import speech_recognition as sr
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from decouple import config
# from openai import OpenAI
# import pydub
# import io
# from pydub.playback import play 
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(audio_data):
#     with open('temp_audio.wav', 'wb') as temp_file:
#         sf.write(temp_file, audio_data.get_raw_data(), 44100)

#     with open('temp_audio.wav', 'rb') as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from the LLM using Langchain with conversation history
# def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE IN ENGLISH LANGUAGE ONLY."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4o")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})  # Corrected input format
    
#     return response.content

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
    
#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = pydub.AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Main script execution
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     print("Press 'Enter' to start recording...")
#     input()  # Wait for the Enter key to start recording

#     transcription = None  # Initialize transcription variable
#     conversation_history = []  # List to keep track of conversation history

#     with microphone as source:
#         print("Recording... Press 'Enter' again to stop.")
#         while True:
#             try:
#                 # Listen for audio and recognize it
#                 audio_data = recognizer.listen(source, timeout=5)
#                 transcription = recognizer.recognize_whisper(audio_data)
#                 print(f"You said: {transcription}")

#                 # Add the user's transcription to the conversation history
#                 conversation_history.append(f"You: {transcription}")

#                 # Get AI response using the conversation history
#                 llm_response = get_openai_llm_response(conversation_history)
#                 print(f"AI says: {llm_response}")

#                 # Add the AI response to the conversation history
#                 conversation_history.append(f"AI: {llm_response}")

#                 # Generate and play speech from LLM response
#                 generate_and_play_speech(llm_response)

#             except sr.WaitTimeoutError:
#                 print("Listening timeout. Press 'Enter' to stop.")
#                 break
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from the service; {e}")
#             except KeyboardInterrupt:
#                 print("Recording stopped.")
#                 break

#     # Final check for transcription
#     if transcription:
#         print("Final transcription was successful.")
#     else:
#         print("No transcription was received.")



# import os
# import soundfile as sf
# import speech_recognition as sr
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from decouple import config
# from openai import OpenAI
# import pydub
# import io
# from pydub.playback import play
# import warnings
# warnings.filterwarnings("ignore", category=FutureWarning)


# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to transcribe audio using OpenAI Whisper
# def transcribe_audio(audio_data):
#     with open('temp_audio.wav', 'wb') as temp_file:
#         sf.write(temp_file, audio_data.get_raw_data(), 44100)

#     with open('temp_audio.wav', 'rb') as audio_file:
#         transcription = client.audio.transcriptions.create(
#             model="whisper-1",
#             file=audio_file,
#             response_format="text",
#             language="en"
#         )
#     return transcription

# # Function to get response from the LLM using Langchain with conversation history
# def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rica who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE IN ENGLISH LANGUAGE ONLY."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4o")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})  # Corrected input format
    
#     return response.content

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
    
#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = pydub.AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Main script execution
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     conversation_history = []  # List to keep track of conversation history
#     listening = False  # Flag to track whether the bot is listening

#     print("Listening for wake word...")  # Initial prompt to user

#     with microphone as source:
#         while True:
#             try:
#                 # Listen for the wake word "Hey Rover"
#                 print("Listening for wake word...")
#                 audio_data = recognizer.listen(source, timeout=5)
#                 wake_word_transcription = recognizer.recognize_whisper(audio_data)

#                 if "hey rover" in wake_word_transcription.lower():
#                     print("Wake word recognized. Starting conversation...")
#                     listening = True
#                     conversation_history.append("System: Wake word activated.")

#                     while listening:
#                         try:
#                             print("Listening for user input...")
#                             audio_data = recognizer.listen(source, timeout=5)
#                             transcription = recognizer.recognize_whisper(audio_data)
#                             print(f"You said: {transcription}")

#                             # Add the user's transcription to the conversation history
#                             conversation_history.append(f"You: {transcription}")

#                             # Get AI response using the conversation history
#                             llm_response = get_openai_llm_response(conversation_history)
#                             print(f"AI says: {llm_response}")

#                             # Add the AI response to the conversation history
#                             conversation_history.append(f"AI: {llm_response}")

#                             # Generate and play speech from LLM response
#                             generate_and_play_speech(llm_response)

#                         except sr.WaitTimeoutError:
#                             print("Listening timeout. You can say 'Hey Rover' again to activate.")
#                             listening = False  # Stop listening for input if timed out
#                         except sr.UnknownValueError:
#                             print("Could not understand audio.")
#                         except sr.RequestError as e:
#                             print(f"Could not request results from the service; {e}")
#                         except KeyboardInterrupt:
#                             print("Recording stopped.")
#                             listening = False  # Stop the conversation loop
#                             break

#                 else:
#                     print("Wake word not recognized. Listening for wake word...")

#             except sr.WaitTimeoutError:
#                 print("Listening timeout. You can say 'Hey Rover' to activate.")
#             except sr.UnknownValueError:
#                 print("Could not understand audio.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from the service; {e}")
#             except KeyboardInterrupt:
#                 print("Exiting the program.")
#                 break

#     # Final check for transcription
#     if transcription:
#         print("Final transcription was successful.")
#     else:
#         print("No transcription was received.")


# import os
# import soundfile as sf
# import speech_recognition as sr
# from io import BytesIO
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_openai import ChatOpenAI
# from decouple import config
# from openai import OpenAI
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Function to transcribe audio using OpenAI Whisper without saving to disk
# def transcribe_audio(audio_data):
#     # Save audio data directly to memory buffer (BytesIO) instead of a file
#     buffer = BytesIO()
#     sf.write(buffer, audio_data.get_raw_data(), 44100, format='WAV')
#     buffer.seek(0)  # Go back to the beginning of the buffer

#     transcription = client.audio.transcriptions.create(
#         model="whisper-small",  # Try `whisper-small` or `whisper-tiny` for faster results
#         file=buffer,
#         response_format="text",
#         language="en"
#     )
#     return transcription

# # Function to get response from the LLM using Langchain
# def get_openai_llm_response(transcribed_text):
#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rica who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE IN ENGLISH LANGUAGE ONLY."),
#             ("human", "{input}"),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4o")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input=transcribed_text)
    
#     return response.content

# # Function to generate speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )

#     # Play the audio (replace with actual audio playing method)
#     for chunk in response.iter_bytes():
#         print(chunk)  # Replace with actual audio playback method

# # Function to listen for the wake word and transcribe audio continuously
# def listen_and_transcribe(recognizer, microphone, wake_word="Hey Rover"):
    
#     # Keep the microphone open
#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#         while True:
#             try:
#                 print("Listening for the wake word...")
                
#                 # Listen for the wake word
#                 audio_data = recognizer.listen(source)
#                 wake_word_transcription = recognizer.recognize_google(audio_data)

#                 if wake_word.lower() in wake_word_transcription.lower():
#                     print(f"Wake word '{wake_word}' detected. Start transcribing...")
                    
#                     while True:
#                         try:
#                             print("Listening for command...")
#                             command_audio = recognizer.listen(source)  # Listen for command
#                             transcription = transcribe_audio(command_audio)
#                             print(f"Transcription: {transcription}")

#                             if transcription.lower() != wake_word.lower():  # Ignore wake word transcription
#                                 llm_response = get_openai_llm_response(transcription)
#                                 print(f"AI says: {llm_response}")
#                                 generate_and_play_speech(llm_response)

#                         except sr.UnknownValueError:
#                             print("Could not understand command audio.")
#                         except sr.RequestError as e:
#                             print(f"Could not request results from service; {e}")
#                         except KeyboardInterrupt:
#                             print("Stopping command listening.")
#                             break
#                 else:
#                     print("Wake word not recognized.")

#             except sr.UnknownValueError:
#                 print("Wake word not recognized.")
#             except sr.RequestError as e:
#                 print(f"Could not request results from service; {e}")

# # Main script execution
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # Start listening for wake word and then transcribe continuously
#     listen_and_transcribe(recognizer, microphone)


# import os
# from decouple import config
# import soundfile as sf
# import speech_recognition as sr
# import numpy as np
# import whisper
# from queue import Queue
# from time import sleep
# from sys import platform
# import torch
# import asyncio
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")


# # Load Whisper model
# def load_whisper_model(model_name="small"):
#     return whisper.load_model(model_name)

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using Langchain
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content  # Return the content of the response

# # Function to transcribe audio stream using Whisper
# async def transcribe_audio_stream(audio_data, model):
#     # Convert raw audio data to NumPy array
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
#     # Run transcription using Whisper
#     result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    
#     return result['text'].strip()

# # Function to handle the conversation logic
# async def handle_conversation(model, recorder, source, record_timeout=2, phrase_timeout=3, energy_threshold=1000):
#     conversation_history = []
#     data_queue = Queue()
    
#     # Callback function for audio recording
#     def record_callback(_, audio: sr.AudioData):
#         data_queue.put(audio.get_raw_data())
    
#     # Set energy threshold for voice detection
#     recorder.energy_threshold = energy_threshold

#     # Start listening in the background
#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    
#     print("Listening for the wake word...")

#     while True:
#         try:
#             # Process audio data from the queue
#             if not data_queue.empty():
#                 audio_data = b''.join(data_queue.queue)
#                 data_queue.queue.clear()

#                 # Transcribe audio using Whisper
#                 transcribed_text = await transcribe_audio_stream(audio_data, model)

#                 # Skip if no meaningful transcription was captured
#                 if not transcribed_text.strip():
#                     continue

#                 print(f"You said: {transcribed_text}")
                
#                 # Add transcribed text to conversation history
#                 conversation_history.append(f"You: {transcribed_text}")
                
#                 # Get LLM response using conversation history
#                 llm_response = await get_openai_llm_response(conversation_history)
#                 print(f"AI says: {llm_response}")
                
#                 # Add LLM response to conversation history
#                 conversation_history.append(f"AI: {llm_response}")

#                 # Play the AI response (you can integrate TTS here)
#                 print(f"Playing response: {llm_response}")
                
#             else:
#                 sleep(0.1)  # Prevent high CPU usage when the queue is empty
#         except KeyboardInterrupt:
#             break

# # Main function to initialize and start conversation
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # Load Whisper model
#     model = load_whisper_model("small")

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#     # Start handling the conversation in real-time
#     asyncio.run(handle_conversation(model, recognizer, microphone))


# import os
# import soundfile as sf
# import speech_recognition as sr
# import numpy as np
# import whisper
# import torch
# import asyncio
# import io
# from queue import Queue
# from time import sleep
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from pydub import AudioSegment
# from pydub.playback import play
# from openai import OpenAI
# from decouple import config
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Load Whisper model
# def load_whisper_model(model_name="small"):
#     return whisper.load_model(model_name)

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )

#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using Langchain
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content  # Return the content of the response

# # Function to transcribe audio stream using Whisper
# async def transcribe_audio_stream(audio_data, model):
#     # Convert raw audio data to NumPy array
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
    
#     # Run transcription using Whisper
#     result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
    
#     return result['text'].strip()

# # Function to handle the conversation logic
# async def handle_conversation(model, recorder, source, record_timeout=2, phrase_timeout=3, energy_threshold=1000):
#     conversation_history = []
#     data_queue = Queue()
    
#     # Callback function for audio recording
#     def record_callback(_, audio: sr.AudioData):
#         data_queue.put(audio.get_raw_data())
    
#     # Set energy threshold for voice detection
#     recorder.energy_threshold = energy_threshold

#     # Start listening in the background
#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)
    
#     print("Listening for the wake word...")

#     wake_word = "hey rover"  # Define your wake word here

#     while True:
#         try:
#             # Process audio data from the queue
#             if not data_queue.empty():
#                 audio_data = b''.join(data_queue.queue)
#                 data_queue.queue.clear()

#                 # Transcribe audio using Whisper
#                 transcribed_text = await transcribe_audio_stream(audio_data, model)

#                 # Skip if no meaningful transcription was captured
#                 if not transcribed_text.strip():
#                     continue

#                 # Check if the wake word is detected
#                 if wake_word.lower() in transcribed_text.lower():
#                     print(f"Wake word detected! Starting conversation...")
#                     while True:
#                         print(f"You said: {transcribed_text}")
                        
#                         # Add transcribed text to conversation history
#                         conversation_history.append(f"You: {transcribed_text}")
                        
#                         # Get LLM response using conversation history
#                         llm_response = await get_openai_llm_response(conversation_history)
#                         print(f"AI says: {llm_response}")
                        
#                         # Add LLM response to conversation history
#                         conversation_history.append(f"AI: {llm_response}")

#                         # Play the AI response using TTS
#                         print(f"Playing response: {llm_response}")
#                         generate_and_play_speech(llm_response)

#                         # Continue conversation
#                         print("Listening for next input...")

#                         # Get new transcribed input after wake word
#                         if not data_queue.empty():
#                             audio_data = b''.join(data_queue.queue)
#                             data_queue.queue.clear()

#                             # Transcribe the new audio input
#                             transcribed_text = await transcribe_audio_stream(audio_data, model)
                            
#                             # Break the loop if user is done talking or presses a key (can add exit keyword)
#                             if "exit" in transcribed_text.lower():
#                                 print("Exiting conversation...")
#                                 break

#                 else:
#                     print("No wake word detected yet...")

#             else:
#                 sleep(0.1)  # Prevent high CPU usage when the queue is empty
#         except KeyboardInterrupt:
#             break

# # Main function to initialize and start conversation
# if __name__ == "__main__":
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     # Load Whisper model
#     model = load_whisper_model("small")

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#     # Start handling the conversation in real-time
#     asyncio.run(handle_conversation(model, recognizer, microphone))


# import os
# import pvporcupine
# import pyaudio
# import speech_recognition as sr
# import asyncio
# import numpy as np
# import torch
# import whisper
# from queue import Queue
# from time import sleep
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from openai import OpenAI
# from pydub import AudioSegment
# from pydub.playback import play
# import io
# from decouple import config
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Initialize Porcupine
# def initialize_porcupine():
#     return pvporcupine.create(keywords=['computer'])  # or use custom wake word

# # Initialize microphone for Porcupine
# def initialize_audio_stream(porcupine):
#     pa = pyaudio.PyAudio()
#     audio_stream = pa.open(
#         rate=porcupine.sample_rate,
#         channels=1,
#         format=pyaudio.paInt16,
#         input=True,
#         frames_per_buffer=porcupine.frame_length
#     )
#     return audio_stream

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )

#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using Langchain
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content  # Return the content of the response

# # Load Whisper model
# def load_whisper_model(model_name="small"):
#     return whisper.load_model(model_name)

# # Function to transcribe audio stream using Whisper
# async def transcribe_audio_stream(audio_data, model):
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#     result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
#     return result['text'].strip()

# # Function to handle the conversation logic
# async def handle_conversation(model, recorder, source, record_timeout=2, phrase_timeout=3, energy_threshold=1000):
#     conversation_history = []
#     data_queue = Queue()

#     # Callback function for audio recording
#     def record_callback(_, audio: sr.AudioData):
#         data_queue.put(audio.get_raw_data())

#     # Set energy threshold for voice detection
#     recorder.energy_threshold = energy_threshold

#     # Start listening in the background
#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

#     print("Listening for input...")

#     while True:
#         try:
#             # Process audio data from the queue
#             if not data_queue.empty():
#                 audio_data = b''.join(data_queue.queue)
#                 data_queue.queue.clear()

#                 # Transcribe audio using Whisper
#                 transcribed_text = await transcribe_audio_stream(audio_data, model)
#                 if transcribed_text:
#                     print(f"You said: {transcribed_text}")
                    
#                     # Add transcribed text to conversation history
#                     conversation_history.append(f"You: {transcribed_text}")

#                     # Get LLM response using conversation history
#                     llm_response = await get_openai_llm_response(conversation_history)
#                     print(f"AI says: {llm_response}")
                    
#                     # Add LLM response to conversation history
#                     conversation_history.append(f"AI: {llm_response}")

#                     # Play the AI response using TTS
#                     print(f"Playing response: {llm_response}")
#                     generate_and_play_speech(llm_response)
#         except KeyboardInterrupt:
#             break

# # Main function to handle the wake word detection and trigger conversation
# def listen_for_wake_word():
#     # Load Whisper model for transcription
#     model = load_whisper_model("small")

#     # Initialize the recognizer and microphone for transcription
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#     # Initialize Porcupine for wake word detection
#     porcupine = initialize_porcupine()
#     audio_stream = initialize_audio_stream(porcupine)

#     print("Listening for the wake word...")

#     try:
#         while True:
#             pcm = audio_stream.read(porcupine.frame_length)
#             pcm = np.frombuffer(pcm, dtype=np.int16)

#             # Check if the wake word is detected
#             if porcupine.process(pcm) >= 0:
#                 print("Wake word detected!")
#                 asyncio.run(handle_conversation(model, recognizer, microphone))

#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         # Clean up
#         audio_stream.close()
#         porcupine.delete()

# if __name__ == "__main__":
#     listen_for_wake_word()


# import os
# import pvporcupine
# import pyaudio
# import speech_recognition as sr
# import asyncio
# import numpy as np
# import torch
# import whisper
# from queue import Queue
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from openai import OpenAI
# from pydub import AudioSegment
# from pydub.playback import play
# import io
# from decouple import config
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Initialize Porcupine
# def initialize_porcupine():
#     return pvporcupine.create(keywords=['computer'])  # or use custom wake word

# # Initialize microphone for Porcupine
# def initialize_audio_stream(porcupine):
#     pa = pyaudio.PyAudio()
#     audio_stream = pa.open(
#         rate=porcupine.sample_rate,
#         channels=1,
#         format=pyaudio.paInt16,
#         input=True,
#         frames_per_buffer=porcupine.frame_length
#     )
#     return audio_stream

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )

#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using Langchain
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content  # Return the content of the response

# # Load Whisper model
# def load_whisper_model(model_name="small"):
#     return whisper.load_model(model_name)

# # Function to transcribe audio stream using Whisper
# async def transcribe_audio_stream(audio_data, model):
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#     result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
#     return result['text'].strip()

# # Function to handle the conversation logic
# async def handle_conversation(model, recorder, source, record_timeout=2, phrase_timeout=3, energy_threshold=1000):
#     conversation_history = []
#     data_queue = Queue()

#     # Callback function for audio recording
#     def record_callback(_, audio: sr.AudioData):
#         data_queue.put(audio.get_raw_data())

#     # Set energy threshold for voice detection
#     recorder.energy_threshold = energy_threshold

#     # Start listening in the background
#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

#     print("Listening for input...")

#     while True:
#         try:
#             # Process audio data from the queue
#             if not data_queue.empty():
#                 audio_data = b''.join(list(data_queue.queue))  # Correctly handle queue data
#                 data_queue.queue.clear()

#                 # Transcribe audio using Whisper
#                 transcribed_text = await transcribe_audio_stream(audio_data, model)
#                 if transcribed_text:
#                     print(f"You said: {transcribed_text}")
                    
#                     # Add transcribed text to conversation history
#                     conversation_history.append(f"You: {transcribed_text}")

#                     # Get LLM response using conversation history
#                     llm_response = await get_openai_llm_response(conversation_history)
#                     print(f"AI says: {llm_response}")
                    
#                     # Add LLM response to conversation history
#                     conversation_history.append(f"AI: {llm_response}")

#                     # Play the AI response using TTS
#                     print(f"Playing response: {llm_response}")
#                     generate_and_play_speech(llm_response)
#         except KeyboardInterrupt:
#             break

# # Main function to handle the wake word detection and trigger conversation
# def listen_for_wake_word():
#     # Load Whisper model for transcription
#     model = load_whisper_model("small")

#     # Initialize the recognizer and microphone for transcription
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#     # Initialize Porcupine for wake word detection
#     porcupine = initialize_porcupine()
#     audio_stream = initialize_audio_stream(porcupine)

#     print("Listening for the wake word...")

#     try:
#         while True:
#             pcm = audio_stream.read(porcupine.frame_length)
#             pcm = np.frombuffer(pcm, dtype=np.int16)

#             # Check if the wake word is detected
#             if porcupine.process(pcm) >= 0:
#                 print("Wake word detected!")
#                 asyncio.run(handle_conversation(model, recognizer, microphone))

#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         # Clean up
#         audio_stream.close()
#         porcupine.delete()

# if __name__ == "__main__":
#     listen_for_wake_word()

# import os
# import pvporcupine
# import pyaudio
# import speech_recognition as sr
# import asyncio
# import numpy as np
# from queue import Queue
# import torch
# import whisper
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from openai import OpenAI
# from pydub import AudioSegment
# from pydub.playback import play
# import io
# from decouple import config
# import warnings

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Initialize Porcupine
# def initialize_porcupine():
#     return pvporcupine.create(keywords=['computer'])  

# # Initialize microphone for Porcupine
# def initialize_audio_stream(porcupine):
#     pa = pyaudio.PyAudio()
#     audio_stream = pa.open(
#         rate=porcupine.sample_rate,
#         channels=1,
#         format=pyaudio.paInt16,
#         input=True,
#         frames_per_buffer=porcupine.frame_length
#     )
#     return audio_stream

# # Function to transcribe audio using OpenAI Whisper
# async def transcribe_audio(audio_data, model):
#     audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
#     result = model.transcribe(audio_np, fp16=torch.cuda.is_available())
#     return result['text'].strip()

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
    
#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content

# # Load Whisper model
# def load_whisper_model(model_name="small"):
#     return whisper.load_model(model_name)

# # Function to handle the conversation logic
# async def handle_conversation(model, recorder, source, record_timeout=2, phrase_timeout=3, energy_threshold=1000):
#     conversation_history = []
#     data_queue = Queue()

#     # Callback function for audio recording
#     def record_callback(_, audio: sr.AudioData):
#         data_queue.put(audio.get_raw_data())

#     # Set energy threshold for voice detection
#     recorder.energy_threshold = energy_threshold

#     # Start listening in the background
#     recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

#     print("Listening for input...")

#     while True:
#         try:
#             # Process audio data from the queue
#             if not data_queue.empty():
#                 audio_data = b''.join(data_queue.queue)
#                 data_queue.queue.clear()

#                 # Transcribe audio using Whisper
#                 transcribed_text = await transcribe_audio(audio_data, model)
#                 if transcribed_text:
#                     print(f"You said: {transcribed_text}")
                    
#                     # Add transcribed text to conversation history
#                     conversation_history.append(f"You: {transcribed_text}")

#                     # Get LLM response using conversation history
#                     llm_response = await get_openai_llm_response(conversation_history)
#                     print(f"AI says: {llm_response}")
                    
#                     # Add LLM response to conversation history
#                     conversation_history.append(f"AI: {llm_response}")

#                     # Play the AI response using TTS
#                     generate_and_play_speech(llm_response)
#         except KeyboardInterrupt:
#             break

# # Main function to handle the wake word detection and trigger conversation
# def listen_for_wake_word():
#     # Load Whisper model for transcription
#     model = load_whisper_model("small")

#     # Initialize the recognizer and microphone for transcription
#     recognizer = sr.Recognizer()
#     microphone = sr.Microphone()

#     with microphone as source:
#         recognizer.adjust_for_ambient_noise(source)

#     # Initialize Porcupine for wake word detection
#     porcupine = initialize_porcupine()
#     audio_stream = initialize_audio_stream(porcupine)

#     print("Listening for the wake word...")

#     try:
#         while True:
#             pcm = audio_stream.read(porcupine.frame_length)
#             pcm = np.frombuffer(pcm, dtype=np.int16)

#             # Check if the wake word is detected
#             if porcupine.process(pcm) >= 0:
#                 print("Wake word detected!")
#                 asyncio.run(handle_conversation(model, recognizer, microphone))

#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         # Clean up
#         audio_stream.close()
#         porcupine.delete()

# if __name__ == "__main__":
#     listen_for_wake_word()



# import os
# import pvporcupine
# import pyaudio
# import asyncio
# import numpy as np
# from queue import Queue
# import torch
# import whisper
# from langchain_openai import ChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from openai import OpenAI
# from pydub import AudioSegment
# from pydub.playback import play
# import io
# from decouple import config
# import warnings
# from RealtimeSTT import AudioToTextRecorder  # Import AudioToTextRecorder for Speech-to-Text

# # Suppress FutureWarnings
# warnings.filterwarnings("ignore", category=FutureWarning)

# # Set up OpenAI API key
# os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
# client = OpenAI()

# # Initialize Porcupine
# def initialize_porcupine():
#     return pvporcupine.create(keywords=['computer'])  # or use custom wake word

# # Initialize microphone for Porcupine
# def initialize_audio_stream(porcupine):
#     pa = pyaudio.PyAudio()
#     audio_stream = pa.open(
#         rate=porcupine.sample_rate,
#         channels=1,
#         format=pyaudio.paInt16,
#         input=True,
#         frames_per_buffer=porcupine.frame_length
#     )
#     return audio_stream

# # Function to generate and play speech using OpenAI TTS model
# def generate_and_play_speech(text_input):
#     response = client.audio.speech.create(
#         model="tts-1",
#         voice="alloy",
#         input=text_input
#     )
    
#     # Play the audio directly from the response
#     audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
#     audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
#     play(audio)  # Play the audio

# # Function to get response from the LLM using Langchain with conversation history
# async def get_openai_llm_response(conversation_history):
#     # Create a formatted conversation string
#     formatted_history = "\n".join(conversation_history)

#     # Define the prompt template using LCEL
#     _prompt = ChatPromptTemplate.from_messages(
#         [
#             ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
#             ("human", formatted_history),
#         ]
#     )

#     # Initialize the OpenAI Chat model (e.g., GPT-4)
#     _model = ChatOpenAI(model="gpt-4")

#     # Chain the prompt with the model using LCEL
#     chain = _prompt | _model

#     # Execute the chain to get the response
#     response = chain.invoke(input={"input": formatted_history})
    
#     return response.content

# # Function to handle the conversation logic using STT from the new method
# async def handle_conversation(recorder):
#     conversation_history = []

#     while True:
#         try:
#             # Capture user input from microphone using AudioToTextRecorder (STT)
#             user_text = recorder.text().strip()

#             if not user_text:
#                 continue

#             print(f"You said: {user_text}")
            
#             # Add transcribed text to conversation history
#             conversation_history.append(f"You: {user_text}")

#             # Get LLM response using conversation history
#             llm_response = await get_openai_llm_response(conversation_history)
#             print(f"AI says: {llm_response}")
            
#             # Add LLM response to conversation history
#             conversation_history.append(f"AI: {llm_response}")

#             # Play the AI response using TTS
#             print(f"Playing response: {llm_response}")
#             generate_and_play_speech(llm_response)

#         except KeyboardInterrupt:
#             break

# # Main function to handle the wake word detection and trigger conversation
# def listen_for_wake_word():
#     # Initialize the recognizer and microphone for transcription
#     recorder = AudioToTextRecorder(
#         model="medium",
#         language="en",
#         wake_words="computer",
#         spinner=True,
#         wake_word_activation_delay=5
#     )

#     # Initialize Porcupine for wake word detection
#     porcupine = initialize_porcupine()
#     audio_stream = initialize_audio_stream(porcupine)

#     print("Listening for the wake word...")

#     try:
#         while True:
#             pcm = audio_stream.read(porcupine.frame_length)
#             pcm = np.frombuffer(pcm, dtype=np.int16)

#             # Check if the wake word is detected
#             if porcupine.process(pcm) >= 0:
#                 print("Wake word detected!")
#                 asyncio.run(handle_conversation(recorder))

#     except KeyboardInterrupt:
#         print("Stopping...")
#     finally:
#         # Clean up
#         audio_stream.close()
#         porcupine.delete()

# if __name__ == "__main__":
#     listen_for_wake_word()


import os
import pvporcupine
import pyaudio
import asyncio
import numpy as np
from queue import Queue
import torch
import whisper
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI
from pydub import AudioSegment
from pydub.playback import play
import io
from decouple import config
import warnings
from RealtimeSTT import AudioToTextRecorder  # Import AudioToTextRecorder for Speech-to-Text

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Set up OpenAI API key
os.environ["OPENAI_API_KEY"] = config("OPEN_AI")
client = OpenAI()

# Initialize Porcupine with a custom wake word .ppn file
def initialize_porcupine():
    access_key = config("PORCUPINE")  # Load the API key from the .env file
    if not access_key:
        raise ValueError("Porcupine access key is missing!")

    return pvporcupine.create(keywords=['computer'])

# Initialize microphone for Porcupine
def initialize_audio_stream(porcupine):
    pa = pyaudio.PyAudio()
    audio_stream = pa.open(
        rate=porcupine.sample_rate,
        channels=1,
        format=pyaudio.paInt16,
        input=True,
        frames_per_buffer=porcupine.frame_length
    )
    return audio_stream

# Function to generate and play speech using OpenAI TTS model
def generate_and_play_speech(text_input):
    response = client.audio.speech.create(
        model="tts-1",
        voice="alloy",
        input=text_input
    )
    
    # Play the audio directly from the response
    audio_data = io.BytesIO(response.content)  # Use BytesIO to handle in-memory audio data
    audio = AudioSegment.from_file(audio_data)  # Load audio data into pydub
    play(audio)  # Play the audio

# Function to get response from the LLM using Langchain with conversation history
async def get_openai_llm_response(conversation_history):
    # Create a formatted conversation string
    formatted_history = "\n".join(conversation_history)

    # Define the prompt template using LCEL
    _prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant. Your name is Rover who only gives one-sentence answers to all questions with a good sense of humor! NO MORE THAN ONE SENTENCE."),
            ("human", formatted_history),
        ]
    )

    # Initialize the OpenAI Chat model (e.g., GPT-4)
    _model = ChatOpenAI(model="gpt-4o-mini")

    # Chain the prompt with the model using LCEL
    chain = _prompt | _model

    # Execute the chain to get the response
    response = chain.invoke(input={"input": formatted_history})
    
    return response.content

# Function to handle the conversation logic using STT from the new method
async def handle_conversation(recorder):
    conversation_history = []

    while True:
        try:
            # Capture user input from microphone using AudioToTextRecorder (STT)
            user_text = recorder.text().strip()

            if not user_text:
                continue

            print(f"You said: {user_text}")
            
            # Add transcribed text to conversation history
            conversation_history.append(f"You: {user_text}")

            # Get LLM response using conversation history
            llm_response = await get_openai_llm_response(conversation_history)
            print(f"AI says: {llm_response}")
            
            # Add LLM response to conversation history
            conversation_history.append(f"AI: {llm_response}")

            # Play the AI response using TTS
            generate_and_play_speech(llm_response)

        except KeyboardInterrupt:
            break

# Main function to handle the wake word detection and trigger conversation
def listen_for_wake_word():
    # Initialize the recognizer and microphone for transcription
    recorder = AudioToTextRecorder(
        model="medium",
        language="en",
        spinner=True,
        wake_word_activation_delay=5
    )

    # Initialize Porcupine for wake word detection with the custom .ppn file
    porcupine = initialize_porcupine()
    audio_stream = initialize_audio_stream(porcupine)

    print("Listening for the wake word...")

    try:
        while True:
            pcm = audio_stream.read(porcupine.frame_length)
            pcm = np.frombuffer(pcm, dtype=np.int16)

            # Check if the wake word is detected
            if porcupine.process(pcm) >= 0:
                print("Wake word detected!")
                asyncio.run(handle_conversation(recorder))

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # Clean up
        audio_stream.close()
        porcupine.delete()

if __name__ == "__main__":
    listen_for_wake_word()
