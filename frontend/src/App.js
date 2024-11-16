//import logo from "./logo.svg";
import "./App.css";
import React, { useState, useEffect, useRef } from "react";
import bowl from "./bowl.svg";
import burrito from "./burrito.svg";
import taco from "./taco.svg";
import quesadilla from "./quesadilla.svg";
import other from "./other.svg";
import axios from "axios";

const baseURL =
  process.env.NODE_ENV === "production"
    ? "https://chipotleaimenu.app"
    : "http://localhost:5000";

const OrderDetails = ({ orderDetails }) => {
  if (!Array.isArray(orderDetails)) {
    return <div>Order is currently empty!</div>;
  }

  const attributeDisplayNames = {
    price: "Price",
    item: "Item",
    meats: "Meat",
    beans: "Beans",
    rice: "Rice",
    toppings: "Toppings",
    completed: "Completed",
    // Add more attributes as needed
  };

  const imageSources = {
    bowl: bowl,
    salad: bowl,
    burrito: burrito,
    taco: taco,
    quesadilla: quesadilla,
    // Add more mappings as needed
  };

  return (
    <div className="">
      {orderDetails.map((order) => {
        const imageSrc = imageSources[order.item] || other;
        return (
          <div key={order._id} className="bg-white rounded-md p-3 ">
            <div className="flex flex-wrap border-b-2 pb-2">
              <div className="relative pr-3">
                <img
                  src={imageSrc}
                  alt={order.name}
                  width="100"
                  height="100"
                  className="rounded-xl pt-1 pl-1"
                />
                <h3 className="absolute top-0 left-0 bg-[#441500] text-white py-1 px-2 rounded-full">
                  {order.order_id}
                </h3>
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 pr-3">
                {Object.keys(attributeDisplayNames).map((key) => {
                  if (key in order && order[key] !== "X") {
                    return (
                      <div key={key} className="flex flex-wrap">
                        <p className="font-bold pr-2">
                          {attributeDisplayNames[key]}:
                        </p>
                        <p>
                          {key === "completed"
                            ? order[key]
                              ? "Yes"
                              : "No"
                            : Array.isArray(order[key])
                            ? order[key].join(", ")
                            : order[key]}
                        </p>
                      </div>
                    );
                  }
                  return null;
                })}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
};

function App() {
  const [messages, setMessages] = useState([]);
  const [userInput, setUserInput] = useState("");
  const [showPopup, setShowPopup] = useState(true);
  const [slideOff, setSlideOff] = useState(false);
  const [sessionId, setSessionId] = useState(null);
  const [orderDetails, setOrderDetails] = useState("");
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const [menuItems, setMenuItems] = useState([]);

  // Initialize sessionId from localStorage when the component mounts
  useEffect(() => {
    const storedSessionId = localStorage.getItem("session_id");
    if (storedSessionId) {
      setSessionId(storedSessionId);
    }
  }, []);

  useEffect(() => {
    const fetchMenuItems = async () => {
      try {
        const response = await axios.get(`${baseURL}/get_menu_items`);
        setMenuItems(response.data.menu_items);
      } catch (error) {
        console.error('Error fetching menu items:', error);
      }
    };

    fetchMenuItems();
  }, []);

  const fetchOrderDetails = async () => {
    try {
      const response = await axios.get(`${baseURL}/get_order`, {
        params: { session_id: sessionId },
      });
      setOrderDetails(response.data.order_details);
      console.log("Order details:", response.data.order_details);
    } catch (error) {
      console.error("Error fetching order details:", error);
    }
  };

  useEffect(() => {
    fetchOrderDetails();
  }, [sessionId]);

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
        session_id: sessionId, // Include session_id in the request
      };

      const response = await axios.post(`${baseURL}/chat`, dataToSend, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      const gptMessage = { text: response.data.response, sender: "chipotle" };
      setMessages((prev) => [...prev, gptMessage]);

      // Store session_id from the response if it's a new session or updated
      if (response.data.session_id && response.data.session_id !== sessionId) {
        setSessionId(response.data.session_id);
        localStorage.setItem("session_id", response.data.session_id);
      }
      await fetchOrderDetails();
    } catch (error) {
      console.error("Error sending message:", error);
      const errorMessage = {
        text: "Sorry, there was an error. Please try again.",
        sender: "chipotle",
      };
      setMessages((prev) => [...prev, errorMessage]);
    }

    setUserInput("");
  };

  const toggleDropdown = () => {
    setIsDropdownOpen(!isDropdownOpen);
  };

  // Scroll to the latest message
  const messagesEndRef = useRef(null);
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="h-screen relative">
      {/* Navbar */}
      <div className="bg-[#441500] w-full py-4 px-6 flex justify-between items-center fixed top-0 left-0 right-0 z-10">
        <div className="text-white text-2xl font-bold font-raleway">Chipotle</div>
        
      </div>

      {/* Popup */}
      {showPopup && (
        <div
          className={`fixed inset-0 bg-black bg-opacity-75 backdrop-blur-md flex justify-center items-center z-20 transition-transform duration-500 ${
            slideOff ? "translate-y-full" : "translate-y-0"
          }`}
        >
          <div className="bg-white font-raleway p-8 rounded-lg shadow-lg flex flex-col items-center justify-center">
            <h2 className="text-4xl font-bold mb-4">Welcome to Chipotle!</h2>
            <p className="mb-4 text-xl">Press continue to start chatting.</p>
            <button
              onClick={handleContinue}
              className="bg-[#AC2318] hover:bg-red-800 text-white text-lg px-4 py-2 rounded-full shadow-lg w-1/2"
            >
              Continue
            </button>
          </div>
        </div>
      )}

      {/* Main Chat Content */}
      <div className="flex flex-col h-screen bg-repeat bg-orange-50 heropattern-topography-orange-100 p-5 z-0 relative pt-20">
        {/* Chat Messages */}
        <div className="flex-1 overflow-auto mb-4">
          {messages.map((msg, idx) => (
            <Message key={idx} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input field */}

        <button
          onClick={toggleDropdown}
          className="mb-2 p-2 bg-[#AC2318] text-white rounded-full shadow-lg"
        >
          {isDropdownOpen ? "Hide Current Order" : "Show Current Order"}
        </button>

        {/* Collapsible dropdown */}
        <div
          className={`collapsible ${
            isDropdownOpen ? "open mb-4 p-2 border border-stone-400 " : ""
          } rounded-2xl shadow-lg px-2`}
        >
          <div>
            <OrderDetails orderDetails={orderDetails} />
          </div>
        </div>
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

function Message({ message }) {
  const { text, sender } = message;
  const [displayedText, setDisplayedText] = useState(
    sender === "chipotle" ? "" : text
  );

  useEffect(() => {
    if (sender === "chipotle") {
      let index = 0;
      const interval = setInterval(() => {
        setDisplayedText(text.slice(0, index + 1));
        index++;
        if (index >= text.length) {
          clearInterval(interval);
        }
      }, 50); // Typing speed for chipotle response
      return () => clearInterval(interval);
    }
  }, [text, sender]);

  return (
    <div
      className={`flex mb-2 ${
        sender === "user" ? "justify-end" : "justify-start"
      }`}
    >
      {sender !== "user" && (
        <div className="flex items-start">
          <div className="flex flex-col items-center">
            <div className="bg-[#441500] text-white rounded-full w-8 h-8 mr-2 flex items-center justify-center mb-1">
              C
            </div>
          </div>
        </div>
      )}
      <div>
        <div className="text-sm text-gray-700 mb-1">
          {sender === "user" ? "" : "ChipotleBot"}
        </div>
        <div
          className={`p-3 rounded-xl max-w-xs text-lg ${
            sender === "user"
              ? "bg-stone-300 text-slate-900 shadow-lg rounded-tr-none"
              : "bg-[#441500] text-white shadow-lg rounded-tl-none"
          }`}
        >
          {displayedText}
        </div>
      </div>
    </div>
  );
}

export default App;
