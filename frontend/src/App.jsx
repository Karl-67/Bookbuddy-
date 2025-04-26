import { useState } from "react";

export default function App() {
  const [isListening, setIsListening] = useState(false);
  const [userMessages, setUserMessages] = useState([]);
  const [aiMessages, setAIMessages] = useState([]);
  const [pronunciationScores, setPronunciationScores] = useState([]);
  const [recordingUrls, setRecordingUrls] = useState([]);
  const [feedbackMessage, setFeedbackMessage] = useState("");

  const playAudio = async (text) => {
    try {
      const response = await fetch("http://localhost:8000/tts", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ text }),
      });

      if (!response.ok) {
        throw new Error("Failed to generate audio");
      }

      // Create a blob from the response
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      // Play the audio
      const audio = new Audio(audioUrl);
      audio.play();

      // Clean up the URL after playing
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
    } catch (err) {
      console.error("Error playing audio:", err);
    }
  };

  const playRecording = async () => {
    try {
      const response = await fetch("http://localhost:8000/get-recording");
      if (!response.ok) {
        throw new Error("Failed to get recording");
      }
      
      const blob = await response.blob();
      const audioUrl = URL.createObjectURL(blob);
      
      const audio = new Audio(audioUrl);
      audio.play();
      
      audio.onended = () => {
        URL.revokeObjectURL(audioUrl);
      };
    } catch (err) {
      console.error("Error playing recording:", err);
    }
  };

  const handlePushToTalk = async () => {
    setIsListening(true);
    setFeedbackMessage("Listening to your speech...");

    try {
      const response = await fetch("http://localhost:8000/analyze", {
        method: "POST",
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "Failed to analyze speech");
      }

      const data = await response.json();
      const userInput = data.transcript;
      const aiResponse = data.simplified;
      const pronunciationScore = data.pronunciation_score;
      const pronunciationStatus = data.pronunciation_status;

      // Update feedback message based on pronunciation score
      if (pronunciationScore < 50) {
        setFeedbackMessage(`Your pronunciation needs improvement (${pronunciationScore.toFixed(1)}%). Try to speak more clearly.`);
      } else if (pronunciationScore < 75) {
        setFeedbackMessage(`Your pronunciation is decent (${pronunciationScore.toFixed(1)}%). Keep practicing!`);
      } else {
        setFeedbackMessage(`Excellent pronunciation! (${pronunciationScore.toFixed(1)}%)`);
      }

      setUserMessages((prev) => [...prev, userInput]);
      setPronunciationScores((prev) => [...prev, 
        { score: pronunciationScore, status: pronunciationStatus }
      ]);

      if (!userInput || userInput.startsWith("❌")) {
        setAIMessages((prev) => [...prev, "❌ Could not detect speech."]);
        setFeedbackMessage("Could not detect speech. Please try again.");
      } else if (!aiResponse || aiResponse.startsWith("❌")) {
        setAIMessages((prev) => [...prev, "❌ Could not simplify. Try again."]);
        setFeedbackMessage("Could not process your speech. Please try again.");
      } else {
        setAIMessages((prev) => [...prev, aiResponse]);
        // Play the simplified text as audio (which includes pronunciation feedback if needed)
        await playAudio(aiResponse);
      }
    } catch (err) {
      console.error("Error during analysis:", err);
      setAIMessages((prev) => [...prev, "❌ Something went wrong."]);
      setFeedbackMessage("Something went wrong. Please try again.");
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
        
        {/* Pronunciation Feedback Banner */}
        {feedbackMessage && (
          <div style={{
            width: "100%",
            padding: "0.75rem",
            marginBottom: "1rem",
            textAlign: "center",
            borderRadius: "8px",
            backgroundColor: "#f0e6cc",
            border: "1px solid #c1b08a",
            boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            fontSize: "1.1rem",
            fontWeight: "500",
            color: "#5a371f"
          }}>
            {feedbackMessage}
          </div>
        )}
        
        <div style={{
          display: "flex",
          width: "100%",
          height: "65vh",
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
              }}>
                {msg}
                {pronunciationScores[i] && (
                  <div>
                    <div style={{
                      marginTop: "0.5rem",
                      padding: "0.25rem 0.5rem",
                      borderRadius: "4px",
                      fontSize: "0.9rem",
                      backgroundColor: pronunciationScores[i].score >= 50 ? "#deefd8" : "#f7d9d9",
                      color: pronunciationScores[i].score >= 50 ? "#2c6e2c" : "#a33c3c",
                      border: `1px solid ${pronunciationScores[i].score >= 50 ? "#a3c8a3" : "#d8a3a3"}`
                    }}>
                      Pronunciation: {pronunciationScores[i].score.toFixed(1)}% - {pronunciationScores[i].status}
                    </div>
                  </div>
                )}
              </div>
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
