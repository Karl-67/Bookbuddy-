import { useState } from "react";

export default function App() {
  const [isListening, setIsListening] = useState(false);
  const [userMessages, setUserMessages] = useState([]);
  const [aiMessages, setAIMessages] = useState([]);

  const speakText = async (text) => {
    try {
      console.log("Sending TTS request with text:", text);
      
      const ttsRes = await fetch("http://localhost:8000/tts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      console.log("TTS Response status:", ttsRes.status);
      console.log("TTS Response headers:", Object.fromEntries(ttsRes.headers.entries()));

      if (!ttsRes.ok) {
        const errorText = await ttsRes.text();
        console.error("TTS request failed:", errorText);
        return;
      }

      // Get the audio blob
      const blob = await ttsRes.blob();
      console.log("Received audio blob:", {
        type: blob.type,
        size: blob.size,
        blob: blob
      });
      
      if (blob.size === 0) {
        console.error("Received empty audio blob");
        return;
      }
      
      // Create a URL for the blob
      const audioUrl = URL.createObjectURL(blob);
      console.log("Created audio URL:", audioUrl);
      
      // Create a new Audio object
      const audio = new Audio(audioUrl);
      
      // Add event listeners for debugging
      audio.onerror = (e) => {
        console.error("Audio playback error:", e);
      };
      
      audio.oncanplaythrough = () => {
        console.log("Audio is ready to play");
      };
      
      audio.onloadeddata = () => {
        console.log("Audio data loaded");
      };
      
      audio.onloadstart = () => {
        console.log("Audio loading started");
      };
      
      // Play the audio
      try {
        const playPromise = audio.play();
        if (playPromise !== undefined) {
          playPromise
            .then(() => {
              console.log("Audio playback started");
            })
            .catch(err => {
              console.error("Error playing audio:", err);
            });
        }
      } catch (err) {
        console.error("Error in audio.play():", err);
      }
      
      // Clean up the URL when done
      audio.onended = () => {
        console.log("Audio playback ended");
        URL.revokeObjectURL(audioUrl);
      };
    } catch (err) {
      console.error("Error in speakText:", err);
    }
  };

  const handlePushToTalk = async () => {
    setIsListening(true);

    try {
      const response = await fetch("http://localhost:8000/transcribe", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        credentials: "include",
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to transcribe");
      }

      const data = await response.json();
      const userInput = data.transcript;
      const aiResponse = data.simplified;

      setUserMessages((prev) => [...prev, userInput]);

      if (!userInput || userInput.startsWith("❌")) {
        setAIMessages((prev) => [...prev, "❌ Could not detect speech."]);
      } else if (!aiResponse || aiResponse.startsWith("❌")) {
        setAIMessages((prev) => [...prev, "❌ Could not simplify. Try again."]);
      } else {
        setAIMessages((prev) => [...prev, aiResponse]);
        speakText(aiResponse); // 🔊 Play AI response
      }
    } catch (err) {
      console.error("Transcription error:", err);
      setAIMessages((prev) => [...prev, `❌ Error: ${err.message}`]);
    } finally {
      setIsListening(false);
    }
  };

  return (
    <div style={{
      height: "100vh",
      width: "100vw",
      background: "radial-gradient(ellipse at center, #ede0c1 0%, #d2b48c 100%)",
      display: "flex",
      justifyContent: "center",
      alignItems: "center",
      fontFamily: "'SF Pro', 'Garamond', 'Georgia', serif",
      color: "#3a2e1e"
    }}>
      <div style={{
        width: "80px",
        height: "100%",
        backgroundColor: "#d6c6a3",
        opacity: 0.5
      }} />
      <div style={{
        flexGrow: 1,
        maxWidth: "1000px",
        height: "90%",
        display: "flex",
        flexDirection: "column",
        justifyContent: "center",
        alignItems: "center",
      }}>
        <h1 style={{
          fontSize: "2.3rem",
          fontWeight: "bold",
          marginBottom: "1rem",
          textShadow: "1px 1px 2px #5e4a2e"
        }}>BookBuddy</h1>
        <div style={{
          display: "flex",
          width: "100%",
          height: "70vh",
          background: "#f7efdb",
          boxShadow: "inset 0 0 20px rgba(0, 0, 0, 0.25)",
          borderRadius: "12px",
          overflow: "hidden",
          border: "5px double #aa9465",
          backdropFilter: "blur(2px)"
        }}>
          {/* Left: User */}
          <div style={{
            flex: 1,
            backgroundColor: "#f4e8c8",
            padding: "1rem",
            overflowY: "auto",
            borderRight: "2px dashed #a78b4e"
          }}>
            <h2 style={{
              textAlign: "center",
              fontWeight: "bold",
              fontSize: "1.25rem",
              marginBottom: "1rem",
              borderBottom: "1px solid #bca474",
              paddingBottom: "0.5rem"
            }}>You Speaketh</h2>
            {userMessages.map((msg, i) => (
              <div key={i} style={{
                padding: "0.75rem",
                margin: "0.5rem 0",
                background: "#efe8d1",
                border: "1px solid #c1b08a",
                borderRadius: "6px"
              }}>{msg}</div>
            ))}
          </div>

          {/* Right: AI */}
          <div style={{
            flex: 1,
            backgroundColor: "#f4e8c8",
            padding: "1rem",
            overflowY: "auto"
          }}>
            <h2 style={{
              textAlign: "center",
              fontWeight: "bold",
              fontSize: "1.25rem",
              marginBottom: "1rem",
              borderBottom: "1px solid #bca474",
              paddingBottom: "0.5rem"
            }}>The Scribe Responds</h2>
            {aiMessages.map((msg, i) => (
              <div key={i} style={{
                padding: "0.75rem",
                margin: "0.5rem 0",
                background: "#efe8d1",
                border: "1px solid #c1b08a",
                borderRadius: "6px"
              }}>{msg}</div>
            ))}
          </div>
        </div>
        <button onClick={handlePushToTalk} style={{
          marginTop: "2rem",
          padding: "1rem 2.5rem",
          borderRadius: "999px",
          fontSize: "1.1rem",
          fontWeight: "bold",
          backgroundColor: "#7c4b2a",
          color: "#fff4d6",
          border: "2px solid #5a371f",
          cursor: "pointer",
          boxShadow: "0 6px 12px rgba(0,0,0,0.3)"
        }}>
          {isListening ? "Listening..." : "Invoke the Voice"}
        </button>
      </div>
      <div style={{
        width: "80px",
        height: "100%",
        backgroundColor: "#d6c6a3",
        opacity: 0.5
      }} />
    </div>
  );
}
