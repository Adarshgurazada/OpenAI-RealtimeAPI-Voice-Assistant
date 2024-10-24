# import asyncio
# import websockets
# import pyaudio
# import numpy as np
# import base64
# import json
# import queue
# import threading
# import os
# from dotenv import load_dotenv
# load_dotenv()

# API_KEY = os.getenv('OPEN_AI')

# # WebSocket URL and header information
# WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
# HEADERS = {
#     "Authorization": "Bearer " + API_KEY,
#     "OpenAI-Beta": "realtime=v1"
# }

# # Initialize queues
# audio_send_queue = queue.Queue()
# audio_receive_queue = queue.Queue()
# conversation_history_id_queue = queue.Queue()

# # Function to convert Base64 to PCM16
# def base64_to_pcm16(base64_audio):
#     return base64.b64decode(base64_audio)

# # Async function to send audio from queue
# async def send_audio_from_queue(websocket):
#     while True:
#         audio_data = await asyncio.get_event_loop().run_in_executor(None, audio_send_queue.get)
#         if audio_data is None:
#             continue
        
#         base64_audio = base64.b64encode(audio_data).decode("utf-8")
#         audio_event = {
#             "type": "input_audio_buffer.append",
#             "audio": base64_audio
#         }

#         await websocket.send(json.dumps(audio_event))
#         await asyncio.sleep(0)

# # Function to read audio from mic and put it into the queue
# def read_audio_to_queue(stream, CHUNK):
#     while True:
#         try:
#             audio_data = stream.read(CHUNK, exception_on_overflow=False)
#             audio_send_queue.put(audio_data)
#         except Exception as e:
#             print(f"Audio reading error: {e}")
#             break

# # Async function to receive audio from server and put it into the queue
# async def receive_audio_to_queue(websocket):
#     print("assistant: ", end="", flush=True)
#     while True:
#         response = await websocket.recv()
#         if response:
#             response_data = json.loads(response)

#             if "type" in response_data and response_data["type"] == "conversation.item.created":
#                 conversation_history_id_queue.put(response_data['item']['id'])

#             if conversation_history_id_queue.qsize() >= 5:
#                 item_id = conversation_history_id_queue.get()
#                 delete_event = {
#                     "type": "conversation.item.delete",
#                     "item_id": item_id
#                 }
#                 await websocket.send(json.dumps(delete_event))
#                 print(f"Deleted conversation_history_id: {item_id}.")

#             if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
#                 print(response_data["delta"], end="", flush=True)
#             elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
#                 print("\nassistant: ", end="", flush=True)
#             elif "type" in response_data and response_data["type"] == "conversation.item.input_audio_transcription.completed":
#                 print("\n↪︎by user messages: ", response_data["transcript"])
#             elif "type" in response_data and response_data["type"] == "rate_limits.updated":
#                 print(f"Rate limits: {response_data['rate_limits'][0]['remaining']} requests remaining.")
#             elif "type" in response_data and response_data["type"] == "response.audio.delta":
#                 base64_audio_response = response_data["delta"]
#                 if base64_audio_response:
#                     pcm16_audio = base64_to_pcm16(base64_audio_response)
#                     audio_receive_queue.put(pcm16_audio)

#         await asyncio.sleep(0)

# # Function to play audio from the queue
# def play_audio_from_queue(output_stream):
#     while True:
#         pcm16_audio = audio_receive_queue.get()
#         if pcm16_audio:
#             output_stream.write(pcm16_audio)

# # Async function for streaming audio and receiving responses
# async def stream_audio_and_receive_response():
#     # WebSocketに接続
#     async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
#         print("WebSocket")

#         update_request = {
#             "type": "session.update",
#             "session": {
#                 "modalities": ["audio", "text"],
#                 "instructions": "Please respond in English with a British accent.",
#                 "voice": "nova", 
#                 "turn_detection": {
#                     "type": "server_vad",
#                     "threshold": 0.5,
#                 },
#                 "input_audio_transcription": {
#                     "model": "whisper-1"
#                 }
#             }
#         }
#         await websocket.send(json.dumps(update_request))

#         # PyAudio setup
#         INPUT_CHUNK = 2400
#         OUTPUT_CHUNK = 2400
#         FORMAT = pyaudio.paInt16
#         CHANNELS = 1
#         INPUT_RATE = 24000
#         OUTPUT_RATE = 24000

#         p = pyaudio.PyAudio()

#         # Initialize microphone stream
#         stream = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=INPUT_CHUNK)

#         # Initialize output stream for server responses
#         output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=OUTPUT_CHUNK)

#         threading.Thread(target=read_audio_to_queue, args=(stream, INPUT_CHUNK), daemon=True).start()
#         threading.Thread(target=play_audio_from_queue, args=(output_stream,), daemon=True).start()

#         try:
#             send_task = asyncio.create_task(send_audio_from_queue(websocket))
#             receive_task = asyncio.create_task(receive_audio_to_queue(websocket))

#             await asyncio.gather(send_task, receive_task)

#         except KeyboardInterrupt:
#             print("Exiting...")
#         finally:
#             stream.stop_stream()
#             stream.close()
#             output_stream.stop_stream()
#             output_stream.close()
#             p.terminate()

# if __name__ == "__main__":
#     asyncio.run(stream_audio_and_receive_response())



import asyncio
import websockets
import pyaudio
import base64
import json
import queue
import threading
import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv('OPEN_AI')

# WebSocket URL and header information
WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
HEADERS = {
    "Authorization": "Bearer " + API_KEY,
    "OpenAI-Beta": "realtime=v1"
}

# Initialize queues
audio_send_queue = queue.Queue()
audio_receive_queue = queue.Queue()
conversation_history_id_queue = queue.Queue()
interrupt_event = threading.Event()  # Event to signal audio interruption
is_playing = threading.Event()  # Event to signal if audio is currently playing

# Function to convert Base64 to PCM16
def base64_to_pcm16(base64_audio):
    return base64.b64decode(base64_audio)

# Async function to send audio from queue
async def send_audio_from_queue(websocket):
    while True:
        audio_data = await asyncio.get_event_loop().run_in_executor(None, audio_send_queue.get)
        if audio_data is None:
            continue
        
        base64_audio = base64.b64encode(audio_data).decode("utf-8")
        audio_event = {
            "type": "input_audio_buffer.append",
            "audio": base64_audio
        }

        await websocket.send(json.dumps(audio_event))
        await asyncio.sleep(0)

# Function to read audio from mic and put it into the queue
def read_audio_to_queue(stream, CHUNK):
    while True:
        try:
            audio_data = stream.read(CHUNK, exception_on_overflow=False)
            audio_send_queue.put(audio_data)
        except Exception as e:
            print(f"Audio reading error: {e}")
            break

# Async function to receive audio from server and put it into the queue
async def receive_audio_to_queue(websocket):
    print("assistant: ", end="", flush=True)
    while True:
        response = await websocket.recv()
        if response:
            response_data = json.loads(response)

            if "type" in response_data and response_data["type"] == "conversation.item.created":
                conversation_history_id_queue.put(response_data['item']['id'])

            if conversation_history_id_queue.qsize() >= 5:
                item_id = conversation_history_id_queue.get()
                delete_event = {
                    "type": "conversation.item.delete",
                    "item_id": item_id
                }
                await websocket.send(json.dumps(delete_event))
                print(f"Deleted conversation_history_id: {item_id}.")

            if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
                print(response_data["delta"], end="", flush=True)
            elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
                print("\nassistant: ", end="", flush=True)
            elif "type" in response_data and response_data["type"] == "conversation.item.input_audio_transcription.completed":
                print("\n↪︎by user messages: ", response_data["transcript"])
                interrupt_event.set()  # Signal that there was a cut-in
            elif "type" in response_data and response_data["type"] == "rate_limits.updated":
                print(f"Rate limits: {response_data['rate_limits'][0]['remaining']} requests remaining.")
            elif "type" in response_data and response_data["type"] == "response.audio.delta":
                base64_audio_response = response_data["delta"]
                if base64_audio_response:
                    pcm16_audio = base64_to_pcm16(base64_audio_response)
                    audio_receive_queue.put(pcm16_audio)

        await asyncio.sleep(0)

# Function to play audio from the queue
def play_audio_from_queue(output_stream):
    while True:
        pcm16_audio = audio_receive_queue.get()
        if pcm16_audio:
            is_playing.set()  # Indicate that audio is playing
            output_stream.write(pcm16_audio)
            if interrupt_event.is_set():  # Check for interruption
                interrupt_event.clear()  # Reset the interrupt event
                is_playing.clear()  # Reset playing status
                break  # Exit after interruption

# Async function for streaming audio and receiving responses
async def stream_audio_and_receive_response():
    async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
        print("WebSocket connected.")

        update_request = {
            "type": "session.update",
            "session": {
                "modalities": ["audio", "text"],
                "instructions": "Please respond in English with a British accent.",
                "voice": "nova", 
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                },
                "input_audio_transcription": {
                    "model": "whisper-1"
                }
            }
        }
        await websocket.send(json.dumps(update_request))

        # PyAudio setup
        INPUT_CHUNK = 2400
        OUTPUT_CHUNK = 2400
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        INPUT_RATE = 24000
        OUTPUT_RATE = 24000

        p = pyaudio.PyAudio()

        # Initialize microphone stream
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=INPUT_CHUNK)

        # Initialize output stream for server responses
        output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=OUTPUT_CHUNK)

        threading.Thread(target=read_audio_to_queue, args=(stream, INPUT_CHUNK), daemon=True).start()
        threading.Thread(target=play_audio_from_queue, args=(output_stream,), daemon=True).start()

        try:
            send_task = asyncio.create_task(send_audio_from_queue(websocket))
            receive_task = asyncio.create_task(receive_audio_to_queue(websocket))

            await asyncio.gather(send_task, receive_task)

        except KeyboardInterrupt:
            print("Exiting...")
        finally:
            stream.stop_stream()
            stream.close()
            output_stream.stop_stream()
            output_stream.close()
            p.terminate()

if __name__ == "__main__":
    asyncio.run(stream_audio_and_receive_response())



# import asyncio
# import websockets
# import pyaudio
# import base64
# import json
# import queue
# import threading
# import os
# from dotenv import load_dotenv

# load_dotenv()

# # Load environment variable for the OpenAI API key
# API_KEY = os.getenv('OPEN_AI')

# # WebSocket URL and header information
# WS_URL = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview-2024-10-01"
# HEADERS = {
#     "Authorization": "Bearer " + API_KEY,
#     "OpenAI-Beta": "realtime=v1"
# }

# # Initialize queues
# audio_send_queue = queue.Queue()
# audio_receive_queue = queue.Queue()

# # Function to convert Base64 to PCM16
# def base64_to_pcm16(base64_audio):
#     return base64.b64decode(base64_audio)

# # Async function to send audio from queue
# async def send_audio_from_queue(websocket):
#     while True:
#         audio_data = await asyncio.get_event_loop().run_in_executor(None, audio_send_queue.get)
#         if audio_data is None:
#             continue
        
#         base64_audio = base64.b64encode(audio_data).decode("utf-8")
#         audio_event = {
#             "type": "input_audio_buffer.append",
#             "audio": base64_audio
#         }
#         await websocket.send(json.dumps(audio_event))
#         await asyncio.sleep(0)

# # Function to read audio from mic and put it into the queue
# def read_audio_to_queue(stream, CHUNK):
#     while True:
#         try:
#             audio_data = stream.read(CHUNK, exception_on_overflow=False)
#             audio_send_queue.put(audio_data)
#         except Exception as e:
#             print(f"Audio reading error: {e}")
#             break

# # Async function to receive audio from server and put it into the queue
# async def receive_audio_to_queue(websocket):
#     print("assistant: ", end="", flush=True)
#     while True:
#         response = await websocket.recv()
#         if response:
#             response_data = json.loads(response)

#             if "type" in response_data and response_data["type"] == "response.audio_transcript.delta":
#                 print(response_data["delta"], end="", flush=True)
#             elif "type" in response_data and response_data["type"] == "response.audio_transcript.done":
#                 print("\nassistant: ", end="", flush=True)
#             elif "type" in response_data and response_data["type"] == "conversation.item.input_audio_transcription.completed":
#                 print("\n↪︎by user messages: ", response_data["transcript"])

#             if "type" in response_data and response_data["type"] == "response.audio.delta":
#                 base64_audio_response = response_data["delta"]
#                 if base64_audio_response:
#                     pcm16_audio = base64_to_pcm16(base64_audio_response)
#                     audio_receive_queue.put(pcm16_audio)

#         await asyncio.sleep(0)

# # Function to play audio from the queue
# def play_audio_from_queue(output_stream):
#     while True:
#         pcm16_audio = audio_receive_queue.get()
#         if pcm16_audio:
#             output_stream.write(pcm16_audio)

# # Async function for streaming audio and receiving responses
# async def stream_audio_and_receive_response():
#     async with websockets.connect(WS_URL, extra_headers=HEADERS) as websocket:
#         print("WebSocket connected.")

#         update_request = {
#             "type": "session.update",
#             "session": {
#                 "modalities": ["audio", "text"],
#                 "instructions": "Please respond in English with a British accent.",
#                 "voice": "nova", 
#                 "turn_detection": {
#                     "type": "server_vad",
#                     "threshold": 0.5,
#                 },
#                 "input_audio_transcription": {
#                     "model": "whisper-1"
#                 }
#             }
#         }
#         await websocket.send(json.dumps(update_request))

#         # PyAudio setup
#         INPUT_CHUNK = 2400
#         OUTPUT_CHUNK = 2400
#         FORMAT = pyaudio.paInt16
#         CHANNELS = 1
#         INPUT_RATE = 24000
#         OUTPUT_RATE = 24000

#         p = pyaudio.PyAudio()
#         stream = p.open(format=FORMAT, channels=CHANNELS, rate=INPUT_RATE, input=True, frames_per_buffer=INPUT_CHUNK)
#         output_stream = p.open(format=FORMAT, channels=CHANNELS, rate=OUTPUT_RATE, output=True, frames_per_buffer=OUTPUT_CHUNK)

#         threading.Thread(target=read_audio_to_queue, args=(stream, INPUT_CHUNK), daemon=True).start()
#         threading.Thread(target=play_audio_from_queue, args=(output_stream,), daemon=True).start()

#         try:
#             send_task = asyncio.create_task(send_audio_from_queue(websocket))
#             receive_task = asyncio.create_task(receive_audio_to_queue(websocket))
#             await asyncio.gather(send_task, receive_task)
#         except KeyboardInterrupt:
#             print("Exiting...")
#         finally:
#             stream.stop_stream()
#             stream.close()
#             output_stream.stop_stream()
#             output_stream.close()
#             p.terminate()

# if __name__ == "__main__":
#     asyncio.run(stream_audio_and_receive_response())
