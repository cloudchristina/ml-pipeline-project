import { useState } from 'react';
import './App.css';

interface PredictionResult {
  predicted_sentiment: string;
  confidence: number;
  probabilities?: {
    POSITIVE: number;
    NEGATIVE: number;
  };
  prediction_time_ms: number;
}

function App() {
  const [text, setText] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

  const handlePredict = async () => {
    if (!text.trim()) {
      setError('Please enter some text');
      return;
    }

    setLoading(true);
    setError('');
    
    try {
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, return_probabilities: true })
      });

      if (!response.ok) throw new Error('Prediction failed');
      
      const data = await response.json();
      setResult(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-600 via-purple-600 to-pink-500 py-12 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-white mb-4">
            🤖 ML Sentiment Analysis
          </h1>
          <p className="text-xl text-white/90">
            Analyze movie review sentiments using DistilBERT
          </p>
        </div>

        <div className="bg-white rounded-2xl shadow-2xl p-8 mb-8">
          <label className="block text-gray-700 text-lg font-semibold mb-4">
            Enter Movie Review
          </label>
          <textarea
            className="w-full h-40 px-4 py-3 border-2 border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none text-gray-800"
            placeholder="Type your movie review here... (e.g., 'This movie was absolutely amazing! Best film I've seen all year.')"
            value={text}
            onChange={(e) => setText(e.target.value)}
          />
          
          <div className="flex gap-4 mt-6">
            <button
              onClick={handlePredict}
              disabled={loading}
              className="flex-1 bg-gradient-to-r from-blue-600 to-purple-600 hover:from-blue-700 hover:to-purple-700 text-white font-bold py-4 px-8 rounded-lg transition-all transform hover:scale-105 disabled:opacity-50 disabled:cursor-not-allowed disabled:transform-none shadow-lg"
            >
              {loading ? '🔄 Analyzing...' : '🚀 Predict Sentiment'}
            </button>
            <button
              onClick={() => { setText(''); setResult(null); setError(''); }}
              className="bg-gray-200 hover:bg-gray-300 text-gray-800 font-bold py-4 px-8 rounded-lg transition-all"
            >
              Clear
            </button>
          </div>

          {error && (
            <div className="mt-6 p-4 bg-red-100 border-l-4 border-red-500 text-red-700 rounded">
              ⚠️ {error}
            </div>
          )}
        </div>

        {result && (
          <div className="bg-white rounded-2xl shadow-2xl p-8 animate-fadeIn">
            <h2 className="text-2xl font-bold text-gray-800 mb-6">Prediction Result</h2>
            
            <div className="text-center mb-8">
              <div className={`inline-block px-8 py-4 rounded-full text-3xl font-bold ${
                result.predicted_sentiment === 'POSITIVE' 
                  ? 'bg-green-100 text-green-700' 
                  : 'bg-red-100 text-red-700'
              }`}>
                {result.predicted_sentiment === 'POSITIVE' ? '😊 POSITIVE' : '😞 NEGATIVE'}
              </div>
            </div>

            <div className="grid md:grid-cols-2 gap-6 mb-6">
              <div className="bg-gradient-to-br from-blue-50 to-blue-100 p-6 rounded-xl">
                <div className="text-sm text-gray-600 mb-2">Confidence</div>
                <div className="text-3xl font-bold text-blue-700">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <div className="mt-3 bg-white/50 rounded-full h-3">
                  <div 
                    className="bg-blue-600 h-3 rounded-full transition-all duration-500"
                    style={{ width: `${result.confidence * 100}%` }}
                  />
                </div>
              </div>

              <div className="bg-gradient-to-br from-purple-50 to-purple-100 p-6 rounded-xl">
                <div className="text-sm text-gray-600 mb-2">Prediction Time</div>
                <div className="text-3xl font-bold text-purple-700">
                  {result.prediction_time_ms.toFixed(1)}ms
                </div>
                <div className="text-sm text-gray-500 mt-2">
                  ⚡ Lightning fast inference
                </div>
              </div>
            </div>

            {result.probabilities && (
              <div className="bg-gray-50 p-6 rounded-xl">
                <h3 className="font-semibold text-gray-700 mb-4">Probability Distribution</h3>
                <div className="space-y-3">
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Negative</span>
                      <span className="font-semibold text-red-600">
                        {(result.probabilities.NEGATIVE * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="bg-white rounded-full h-6">
                      <div 
                        className="bg-red-500 h-6 rounded-full flex items-center justify-end pr-2 text-xs text-white font-semibold transition-all duration-500"
                        style={{ width: `${result.probabilities.NEGATIVE * 100}%` }}
                      >
                        {result.probabilities.NEGATIVE > 0.1 && '😞'}
                      </div>
                    </div>
                  </div>
                  <div>
                    <div className="flex justify-between text-sm mb-1">
                      <span className="text-gray-600">Positive</span>
                      <span className="font-semibold text-green-600">
                        {(result.probabilities.POSITIVE * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="bg-white rounded-full h-6">
                      <div 
                        className="bg-green-500 h-6 rounded-full flex items-center justify-end pr-2 text-xs text-white font-semibold transition-all duration-500"
                        style={{ width: `${result.probabilities.POSITIVE * 100}%` }}
                      >
                        {result.probabilities.POSITIVE > 0.1 && '😊'}
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            )}
          </div>
        )}

        <div className="mt-8 text-center text-white/80 text-sm">
          <p>Powered by DistilBERT • FastAPI Backend • React Frontend</p>
          <p className="mt-2">🔗 API: <code className="bg-white/20 px-2 py-1 rounded">http://localhost:8000</code></p>
        </div>
      </div>
    </div>
  );
}

export default App;
