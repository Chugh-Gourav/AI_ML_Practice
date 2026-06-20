import React, { useState, useRef, useEffect } from 'react';
import { Send, Sparkles, Moon, Star, MapPin, User, Loader2, ChevronDown } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { motion, AnimatePresence } from 'framer-motion';

function App() {
  const [messages, setMessages] = useState([
    { role: 'agent', content: "Greetings! ✨ I'm your Zodiac Travel Guide. Tell me your budget and vibe, and let's find your perfect destination!" }
  ]);
  const [input, setInput] = useState('');
  const [userId, setUserId] = useState('user_001'); // Default to first user
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim()) return;

    // Simplified logic: We rely on the dropdown now.
    let currentUserId = userId;
    let messageToSend = input;

    // Fallback if somehow empty (shouldn't be with default state)
    if (!currentUserId) {
      currentUserId = "user_001";
      setUserId("user_001");
    }

    const newMessages = [...messages, { role: 'user', content: input }];
    setMessages(newMessages);
    setInput('');
    setIsLoading(true);

    const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

    try {
      const response = await fetch(`${API_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: currentUserId,
          message: messageToSend,
          history: newMessages // Pass full history
        }),
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      setMessages([...newMessages, { role: 'agent', content: data.response }]);
    } catch (error) {
      console.error('Error:', error);
      setMessages([...newMessages, { role: 'agent', content: "I'm having trouble connecting to the stars right now. Please try again later." }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="min-h-screen bg-[#02122c] text-white font-sans overflow-hidden relative selection:bg-[#0770e3] selection:text-white">
      {/* Background Stars - Subtle */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        {[...Array(20)].map((_, i) => (
          <div
            key={i}
            className="absolute rounded-full bg-white opacity-20 animate-pulse"
            style={{
              top: `${Math.random() * 100}%`,
              left: `${Math.random() * 100}%`,
              width: `${Math.random() * 3 + 1}px`,
              height: `${Math.random() * 3 + 1}px`,
              animationDelay: `${Math.random() * 5}s`
            }}
          />
        ))}
      </div>

      <div className="container mx-auto px-4 h-screen flex flex-col max-w-4xl relative z-10">
        {/* Header */}
        <header className="py-6 flex items-center justify-between border-b border-white/10">
          <div className="flex items-center gap-3">
            <div className="p-2 bg-[#0770e3] rounded-lg shadow-lg shadow-[#0770e3]/20">
              <Sparkles className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold text-white tracking-tight">
                Zodiac Travel
              </h1>
              <p className="text-sm text-sky-200 font-medium">Let your stars take you somewhere</p>
            </div>
          </div>
          <div className="relative">
            <div className="flex items-center gap-2 text-sm text-white bg-white/10 py-1.5 px-4 rounded-full border border-white/10 font-medium hover:bg-white/20 transition-colors cursor-pointer">
              <User className="w-4 h-4" />
              <select
                value={userId}
                onChange={(e) => setUserId(e.target.value)}
                className="bg-transparent border-none focus:ring-0 text-white text-sm font-medium appearance-none cursor-pointer pr-4 [&>option]:bg-[#02122c]"
              >
                <option value="" disabled>Select User</option>
                <option value="user_001">Logged in 01</option>
                <option value="user_002">Logged in 02</option>
                <option value="user_003">Logged in 03</option>
              </select>
              <ChevronDown className="w-3 h-3 absolute right-3 pointer-events-none opacity-70" />
            </div>
          </div>
        </header>

        {/* Chat Area */}
        <main className="flex-1 overflow-y-auto p-4 space-y-6 my-4 scrollbar-thin scrollbar-thumb-[#0770e3]/50 scrollbar-track-transparent">
          <AnimatePresence>
            {messages.map((msg, idx) => (
              <motion.div
                key={idx}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
              >
                <div
                  className={`max-w-[80%] rounded-2xl p-4 shadow-lg backdrop-blur-sm border ${msg.role === 'user'
                    ? 'bg-[#0770e3] border-[#0770e3] text-white rounded-br-none'
                    : 'bg-[#111f38] border-white/10 text-sky-50 rounded-bl-none'
                    }`}
                >
                  <div className="flex items-center gap-2 mb-1 opacity-70 text-xs uppercase tracking-wide font-bold">
                    {msg.role === 'user' ? <User className="w-3 h-3" /> : <Moon className="w-3 h-3" />}
                    {msg.role === 'user' ? 'You' : 'Travel Recommendation'}
                  </div>
                  <div className="prose prose-invert prose-sm">
                    <ReactMarkdown>{msg.content}</ReactMarkdown>
                  </div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>

          {isLoading && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex justify-start"
            >
              <div className="bg-[#111f38] border border-white/10 text-sky-50 rounded-2xl p-4 rounded-bl-none flex items-center gap-3">
                <Loader2 className="w-4 h-4 animate-spin text-[#0770e3]" />
                <span className="text-sm font-medium">Consulting the stars...</span>
              </div>
            </motion.div>
          )}
          <div ref={messagesEndRef} />
        </main>

        {/* Input Area */}
        <footer className="py-6 pt-2">
          <div className="relative group">
            <div className="absolute -inset-0.5 bg-[#0770e3] rounded-xl blur opacity-20 group-hover:opacity-40 transition duration-500"></div>
            <div className="relative flex items-center bg-[#02122c] rounded-xl border border-white/20 shadow-2xl overflow-hidden focus-within:border-[#0770e3] transition-colors">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder="Where would you like to explore next?"
                className="flex-1 bg-transparent border-none px-6 py-4 text-white placeholder-slate-400 focus:outline-none focus:ring-0 font-medium"
              />
              <button
                onClick={handleSend}
                disabled={isLoading}
                className="px-6 py-4 bg-[#0770e3] hover:bg-[#055bb5] text-white transition-colors disabled:opacity-50 disabled:cursor-not-allowed font-bold"
              >
                <Send className="w-5 h-5" />
              </button>
            </div>
          </div>
          <p className="text-center text-slate-500 text-xs mt-3">
            Powered by Vertex AI Agent Engine
          </p>
        </footer>
      </div>
    </div>
  );
}

export default App;
