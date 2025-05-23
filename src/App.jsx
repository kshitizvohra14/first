import React, { useRef, useState, useEffect } from 'react';
import Webcam from 'react-webcam';
import axios from 'axios';

const SignDetection = () => {
  const webcamRef = useRef(null);
  const [prediction, setPrediction] = useState('');
  const [loading, setLoading] = useState(false);
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const interval = setInterval(async () => {
      if (
        webcamRef.current &&
        webcamRef.current.getScreenshot &&
        !loading
      ) {
        const imageSrc = webcamRef.current.getScreenshot();
        if (imageSrc) {
          setLoading(true);
          try {
            const response = await axios.post('http://localhost:5000/predict', {
              image: imageSrc
            });
            const result = response.data.prediction;
            setPrediction(result);
            setHistory((prev) => [result, ...prev.slice(0, 4)]); // Keep last 5

            // 🔊 Audio feedback
            const utterance = new SpeechSynthesisUtterance(result);
            speechSynthesis.speak(utterance);
          } catch (error) {
            console.error("Prediction failed", error);
            setPrediction("Error predicting sign.");
          }
          setLoading(false);
        }
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [loading]);

  return (
    <div className="flex h-screen bg-gray-100">
      {/* Sidebar */}
      <aside className="w-64 bg-white shadow-lg px-4 py-6 hidden sm:block">
        <h2 className="text-xl font-bold text-gray-800 mb-6">Dashboard</h2>
        <nav className="flex flex-col gap-4 text-gray-600">
          <a href="#" className="hover:text-blue-600">Home</a>
          <a href="#" className="hover:text-blue-600">Sign Detection</a>
          <a href="#" className="hover:text-blue-600">Training Module</a>
          <a href="#" className="hover:text-blue-600">History</a>
          <a href="#" className="hover:text-blue-600">Settings</a>
        </nav>
      </aside>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Navbar */}
        <header className="bg-white shadow-md px-6 py-4 flex items-center justify-between">
          <h1 className="text-xl font-bold text-gray-800">GESTURE WORDS</h1>
          <button className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700 transition">
            Login
          </button>
        </header>

        {/* Content */}
        <main className="flex flex-col items-center justify-center flex-1 p-6">
          <div className="bg-white rounded-xl shadow-md p-6 w-full max-w-md text-center">
            <h2 className="text-xl font-semibold text-gray-700 mb-4">Live Sign Detection</h2>

            <div className="border-4 border-gray-200 rounded-lg overflow-hidden w-64 mx-auto mb-4">
              <Webcam
                audio={false}
                ref={webcamRef}
                screenshotFormat="image/jpeg"
                width={256}
                className="rounded"
              />
            </div>

            <p className="text-sm text-gray-500 mb-2">{loading ? "Analyzing..." : "Waiting for next frame..."}</p>

            {prediction && (
              <div className="mt-2">
                <h3 className="text-lg text-gray-600">Prediction:</h3>
                <p className="text-green-600 text-3xl font-bold">{prediction}</p>
              </div>
            )}

            {history.length > 0 && (
              <div className="mt-4 text-left">
                <h4 className="text-sm font-semibold text-gray-500 mb-1">Recent Predictions:</h4>
                <ul className="text-gray-700 text-sm space-y-1">
                  {history.map((item, index) => (
                    <li key={index} className="border p-1 rounded bg-gray-100">{item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

export default SignDetection;
