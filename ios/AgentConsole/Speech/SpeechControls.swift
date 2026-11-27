import SwiftUI

/// Tap to talk, tap again to stop: the transcript is sent to the agent.
struct SpeechMicButton: View {
    let onTranscript: (String) -> Void
    private var speech: SpeechService { .shared }

    var body: some View {
        Button {
            Task {
                if speech.isRecording {
                    let text = await speech.stopRecordingAndTranscribe()
                    if !text.isEmpty { onTranscript(text) }
                } else {
                    await speech.startRecording()
                }
            }
        } label: {
            Label(speech.isRecording ? "Stop recording" : "Speak",
                  systemImage: speech.isRecording ? "mic.circle.fill" : "mic.circle")
                .foregroundStyle(speech.isRecording ? Color.red : Color.accentColor)
                .symbolEffect(.pulse, isActive: speech.isRecording)
        }
        .disabled(speech.isBusy && !speech.isRecording)
    }
}

/// Toolbar toggle: read assistant replies aloud.
struct SpeakerToggle: View {
    var body: some View {
        @Bindable var speech = SpeechService.shared
        Toggle(isOn: $speech.speakReplies) {
            Label("Read replies aloud",
                  systemImage: speech.speakReplies ? "speaker.wave.2.fill" : "speaker.slash")
        }
        .toggleStyle(.button)
    }
}

/// Status line for downloads (first use needs network), loading, transcription, errors.
struct SpeechStatusBar: View {
    private var speech: SpeechService { .shared }

    var body: some View {
        Group {
            switch speech.phase {
            case .downloading(let model, let p):
                ProgressView("Downloading \(model) (first use, needs network)…", value: p)
            case .loading(let model):
                ProgressView("Loading \(model)…")
            case .recording:
                Label("Listening… tap the mic again to send", systemImage: "waveform")
            case .transcribing:
                ProgressView("Transcribing…")
            case .synthesizing, .speaking, .idle:
                if let err = speech.lastError {
                    Label(err, systemImage: "exclamationmark.triangle").foregroundStyle(.orange)
                }
            }
        }
        .font(.footnote)
        .padding(.horizontal)
    }
}

extension View {
    /// Speaks each finished assistant reply when the speaker toggle is on.
    func speechReadsReplies(_ agent: LocalAgent) -> some View {
        onChange(of: agent.isRunning) { _, running in
            if running {
                SpeechService.shared.stopSpeaking()
            } else if let last = agent.turns.last, last.role == .assistant {
                SpeechService.shared.speakIfEnabled(last.text)
            }
        }
    }
}
