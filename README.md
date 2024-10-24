
# Interruptible Voice Assistant Using OpenAI RealTime API and Wake Word Detection

This repository contains two Python scripts that work together to create an interruptible voice assistant powered by OpenAI's RealTime API. The assistant features wake word detection, real-time speech-to-text (STT), text-to-speech (TTS), and the ability to carry out conversations using OpenAI’s GPT model.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup Instructions](#setup-instructions)
  - [Environment Variables](#environment-variables)
  - [Requirements](#requirements)
  - [Running the Code](#running-the-code)
- [Code Breakdown](#code-breakdown)
  - [REALTIMEAPI.py](#realtimeapipy)
  - [voice.py](#voicepy)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This repository is designed to handle voice interactions with an assistant that:
1. Listens for a wake word (such as "computer").
2. Processes speech in real-time and returns AI-generated responses.
3. Uses both OpenAI's Whisper model for speech recognition and OpenAI's TTS models for generating human-like responses.

---

## Features
- **Real-time Speech Recognition (STT)**: Streams audio from the microphone to OpenAI's Whisper model for transcription.
- **Wake Word Detection**: Uses Picovoice's Porcupine library for efficient wake word detection.
- **Text-to-Speech (TTS)**: Generates real-time audio responses using OpenAI's TTS model.
- **Interruptible Responses**: Allows the user to interrupt the assistant while it is speaking.
- **Conversation Management**: Tracks and deletes conversation history dynamically.

---

## Setup Instructions

### Environment Variables

Create a `.env` file in the root of the project and add the following variables:

```bash
OPEN_AI=your_openai_api_key
PORCUPINE=your_porcupine_access_key
```

### Requirements

To install the required Python packages, run:

```bash
pip install -r requirements.txt
```

Make sure you also have `pyaudio` installed, which may require system-level dependencies. For example, on Debian-based systems:

```bash
sudo apt-get install portaudio19-dev
```

### Running the Code

1. Clone the repository:
    ```bash
    git clone https://github.com/Adarshgurazada/OpenAI-RealtimeAPI-Voice-Assistant.git
    cd voice-assistant
    ```

2. Install the dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the voice assistant scripts. You can start either script depending on your needs:

    - **To start the assistant with OpenAI RealTime API:**
      ```bash
      python REALTIMEAPI.py
      ```

    - **To run the assistant with wake word detection:**
      ```bash
      python voice.py
      ```

---

## Code Breakdown

### REALTIMEAPI.py

The `REALTIMEAPI.py` script connects to OpenAI’s RealTime API to perform real-time speech recognition and response. It streams audio from your microphone, processes it through OpenAI’s Whisper model for transcription, and plays the response using text-to-speech.

Key features:
- **Wake Word Detection**: Not included.
- **Real-time Speech-to-Text**: Uses OpenAI’s Whisper model.
- **Text-to-Speech**: Responds in English with a British accent.

### voice.py

The `voice.py` script uses Picovoice's Porcupine for wake word detection, allowing you to trigger the assistant by saying a specific keyword. Once activated, it uses OpenAI for generating responses based on the user's input.

Key features:
- **Wake Word Detection**: Triggered by the keyword "computer" using the Porcupine library.
- **STT and LLM**: Uses OpenAI's Whisper model for speech-to-text and Langchain for handling conversation logic.
- **Text-to-Speech**: Plays the response using OpenAI’s TTS model.

---

## Contributing

Feel free to fork the project, submit issues, or make pull requests.

---

## License

This project is licensed under the MIT License.
