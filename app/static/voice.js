/* ============================================================
   RAG Terminal — voice.js
   Web Speech API integration (STT & TTS)
   ============================================================ */

'use strict';

window.VoiceManager = {
  recognition: null,
  isListening: false,
  ttsEnabled: false,
  onResultCallback: null,
  onEndCallback: null,

  init() {
    // Initialize SpeechRecognition
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (SpeechRecognition) {
      this.recognition = new SpeechRecognition();
      this.recognition.continuous = false; // We want one-shot by default for commands
      this.recognition.interimResults = true; // Show text as user speaks
      this.recognition.lang = 'en-US';

      this.recognition.onstart = () => {
        this.isListening = true;
      };

      this.recognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        if (this.onResultCallback) {
          this.onResultCallback(interimTranscript, finalTranscript);
        }
      };

      this.recognition.onerror = (event) => {
        console.error('Speech recognition error', event.error);
        if (event.error === 'not-allowed') {
           printError("Microphone access denied. Please allow microphone permissions.");
        }
        this.stopListening();
      };

      this.recognition.onend = () => {
        this.isListening = false;
        if (this.onEndCallback) {
          this.onEndCallback();
        }
      };
    } else {
      console.warn("Speech Recognition API not supported in this browser.");
    }
  },

  toggleListening(onResult, onEnd) {
    if (!this.recognition) {
      printError("Speech Recognition is not supported in your browser. Try Chrome/Edge/Safari.");
      return false;
    }

    if (this.isListening) {
      this.stopListening();
      return false;
    } else {
      this.onResultCallback = onResult;
      this.onEndCallback = onEnd;
      try {
        this.recognition.start();
        return true;
      } catch (err) {
        console.error(err);
        return false;
      }
    }
  },

  stopListening() {
    if (this.recognition && this.isListening) {
      this.recognition.stop();
    }
  },

  // --- TTS ---

  toggleTTS() {
    this.ttsEnabled = !this.ttsEnabled;
    if (!this.ttsEnabled) {
      this.stopSpeaking();
    }
    return this.ttsEnabled;
  },

  speak(text) {
    if (!this.ttsEnabled || !window.speechSynthesis) return;

    // Stop current speech
    this.stopSpeaking();

    // Chunk by sentences to avoid TTS cutoffs and sound more natural
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [text];
    
    sentences.forEach((sentence) => {
      const utterance = new SpeechSynthesisUtterance(sentence.trim());
      utterance.rate = 1.1;
      utterance.pitch = 0.9; // Slightly lower pitch for terminal feel
      window.speechSynthesis.speak(utterance);
    });
  },

  stopSpeaking() {
    if (window.speechSynthesis) {
        window.speechSynthesis.cancel();
    }
  }
};

// Initialize early
window.VoiceManager.init();
