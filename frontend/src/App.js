//import logo from "./logo.svg";
import "./App.css";
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");

  const [showPopup, setShowPopup] = useState(true);
  const [slideOff, setSlideOff] = useState(false);

  // Initialize sessionId from localStorage
  const [sessionId, setSessionId] = useState(() => {
    return localStorage.getItem("session_id");
  });

  const handleContinue = () => {
    setSlideOff(true); // Start the slide animation
    setTimeout(() => {
      setShowPopup(false); // Remove the popup after the animation completes
    }, 1000); // Match this with the duration of the slide animation
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!userInput.trim()) {
      return; // Don't send empty messages
    }

    const userMessage = { text: userInput, sender: "user" };
    setMessages((prev) => [...prev, userMessage]);

    try {
      // Prepare the data to send
      const dataToSend = {
        message: userInput,
      };

      // Include session_id if it's already available
      if (sessionId) {
        dataToSend.session_id = sessionId;
      }

      const response = await axios.post("http://localhost:5000/chat", dataToSend);

      const gptMessage = { text: response.data.response, sender: "chipotle" };
      setMessages((prev) => [...prev, gptMessage]);

      // Store session_id from the response if it's a new session or updated
      if (response.data.session_id && response.data.session_id !== sessionId) {
        setSessionId(response.data.session_id);
        localStorage.setItem("session_id", response.data.session_id);
      }
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = { text: "Sorry, there was an error. Please try again.", sender: "chipotle" };
      setMessages((prev) => [...prev, errorMessage]);
    }

    setUserInput("");
  };

  return (
    <div className="h-screen relative">
      {/* Navbar */}
      <div className="bg-[#441500] w-full py-4 px-6 flex justify-between items-center fixed top-0 left-0 right-0 z-10">
        <div className="text-white text-2xl font-bold">Chipotle</div>
      </div>

      {/* Popup */}
      {showPopup && (
        <div
          className={`fixed inset-0 bg-black bg-opacity-75 backdrop-blur-md flex justify-center items-center z-20 transition-transform duration-500 ${
            slideOff ? "translate-y-full" : "translate-y-0"
          }`}
        >
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <h2 className="text-2xl font-bold mb-4">
              Welcome to Chipotle Chat!
            </h2>
            <p className="mb-4">Press continue to start chatting.</p>
            <button
              onClick={handleContinue}
              className="bg-[#AC2318] text-white px-4 py-2 rounded-lg shadow-lg"
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {/* Main Chat Content */}
      <div className="flex flex-col h-screen bg-repeat heropattern-topography-stone-100 p-5 z-0 relative pt-20">
        {/* Chat Messages */}
        <div className="flex-1 overflow-auto mb-4">
          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex mb-2 ${
                msg.sender === "user" ? "justify-end" : "justify-start"
              }`}
            >
              <div
                className={`p-3 rounded-xl max-w-xs text-lg ${
                  msg.sender === "user"
                    ? "bg-slate-200 text-slate-900 shadow-lg"
                    : "bg-[#441500] text-white shadow-lg"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
        </div>

        {/* Input field */}
        <form onSubmit={handleSubmit} className="flex">
          <input
            type="text"
            value={userInput}
            onChange={(e) => setUserInput(e.target.value)}
            className="flex-1 p-2 ph-4 border border-stone-400 rounded-2xl text-2xl shadow-lg"
            placeholder="Type a message..."
          />
          <button
            type="submit"
            className="ml-2 p-2 bg-[#AC2318] text-white rounded-full shadow-lg"
          >
            <svg
              fill="#FFF"
              className="w-4 h-4 sm:w-8 sm:h-4 md:w-10 md:h-6 lg:w-12 lg:h-8"
              xmlns="http://www.w3.org/2000/svg"
              viewBox="0 0 495.003 495.003"
            >
              <g id="XMLID_51_">
                <path
                  id="XMLID_53_"
                  d="M164.711,456.687c0,2.966,1.647,5.686,4.266,7.072c2.617,1.385,5.799,1.207,8.245-0.468l55.09-37.616
                l-67.6-32.22V456.687z"
                />
                <path
                  id="XMLID_52_"
                  d="M492.431,32.443c-1.513-1.395-3.466-2.125-5.44-2.125c-1.19,0-2.377,0.264-3.5,0.816L7.905,264.422
                c-4.861,2.389-7.937,7.353-7.904,12.783c0.033,5.423,3.161,10.353,8.057,12.689l125.342,59.724l250.62-205.99L164.455,364.414
                l156.145,74.4c1.918,0.919,4.012,1.376,6.084,1.376c1.768,0,3.519-0.322,5.186-0.977c3.637-1.438,6.527-4.318,7.97-7.956
                L494.436,41.257C495.66,38.188,494.862,34.679,492.431,32.443z"
                />
              </g>
            </svg>
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
