import React, { useState, useRef, useEffect } from "react";
import { sendChat } from "../api/cogniClient";

function Message({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div
        className={`w-8 h-8 rounded-xl flex items-center justify-center text-sm font-bold shrink-0 ${
          isUser ? "bg-cogni-500 text-white" : "bg-gray-700 text-gray-300"
        }`}
      >
        {isUser ? "M" : "AI"}
      </div>
      <div className={`max-w-[80%] space-y-2 ${isUser ? "items-end" : "items-start"} flex flex-col`}>
        <div
          className={`rounded-2xl px-4 py-3 text-sm leading-relaxed ${
            isUser
              ? "bg-cogni-500 text-white rounded-tr-sm"
              : "bg-gray-800 text-gray-200 rounded-tl-sm"
          }`}
        >
          {msg.content}
        </div>

        {/* Recommendations */}
        {!isUser && msg.recommendations?.length > 0 && (
          <div className="w-full space-y-2 mt-2">
            <div className="text-xs text-gray-500 font-semibold uppercase tracking-wide">Recommended Actions</div>
            {msg.recommendations.map((rec, i) => (
              <div key={i} className="bg-gray-900 border border-gray-700 rounded-xl p-3 text-sm">
                <div className="font-semibold text-white mb-1">{i + 1}. {rec.what}</div>
                <div className="text-gray-400 text-xs mb-1">{rec.why}</div>
                <div className="text-xs text-gray-500">
                  Timeline: <span className="text-gray-300">{rec.timeline}</span>
                </div>
                {rec.script && (
                  <div className="mt-2 italic text-xs text-cogni-400 border-l-2 border-cogni-500/50 pl-2">
                    "{rec.script}"
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Sources */}
        {!isUser && msg.sources?.length > 0 && (
          <div className="text-xs text-gray-600">
            Sources: {msg.sources.map((s) => s.collection).join(", ")}
          </div>
        )}

        {/* Risk badge */}
        {!isUser && msg.risk_level && (
          <span
            className={`text-xs px-2 py-0.5 rounded-full font-medium ${
              msg.risk_level === "critical"
                ? "bg-red-900/50 text-red-400"
                : msg.risk_level === "high"
                ? "bg-orange-900/50 text-orange-400"
                : msg.risk_level === "medium"
                ? "bg-yellow-900/50 text-yellow-400"
                : "bg-green-900/50 text-green-400"
            }`}
          >
            Risk: {msg.risk_level}
          </span>
        )}
      </div>
    </div>
  );
}

const SUGGESTED_QUESTIONS = [
  "Who on my team is at risk of burnout this week?",
  "Are there any conflict signals I should be aware of?",
  "Which employees are most likely to leave in the next 90 days?",
  "What's the overall sentiment trend for my team?",
];

export default function Chat() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      content:
        "Hi! I'm your CogniTeam AI assistant. I can help you understand your team's health, identify risks, and suggest actions. What would you like to know?",
    },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async (question) => {
    const q = question || input.trim();
    if (!q || loading) return;

    setMessages((prev) => [...prev, { role: "user", content: q }]);
    setInput("");
    setLoading(true);

    try {
      const data = await sendChat(q);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || data.interpretation || "I couldn't generate a response.",
          recommendations: data.recommendations || [],
          sources: data.sources || [],
          risk_level: data.risk_level,
        },
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: `Error: ${err.response?.data?.detail || "Something went wrong. Please try again."}`,
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen max-h-screen">
      {/* Header */}
      <div className="px-8 py-5 border-b border-gray-800 bg-gray-950">
        <h1 className="text-xl font-bold text-white">AI Team Assistant</h1>
        <p className="text-gray-500 text-sm mt-0.5">Powered by LLaMA 3 · Responses are privacy-protected</p>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto px-8 py-6 space-y-6">
        {messages.map((msg, i) => (
          <Message key={i} msg={msg} />
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-xl bg-gray-700 flex items-center justify-center text-sm text-gray-300 font-bold">
              AI
            </div>
            <div className="bg-gray-800 rounded-2xl rounded-tl-sm px-4 py-3">
              <div className="flex gap-1 items-center">
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <div className="w-1.5 h-1.5 bg-gray-500 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      {/* Suggested questions */}
      {messages.length <= 1 && (
        <div className="px-8 pb-4">
          <div className="text-xs text-gray-600 mb-2 font-medium">Suggested questions</div>
          <div className="flex flex-wrap gap-2">
            {SUGGESTED_QUESTIONS.map((q) => (
              <button
                key={q}
                onClick={() => handleSend(q)}
                className="text-xs text-gray-400 bg-gray-800 hover:bg-gray-700 px-3 py-1.5 rounded-xl border border-gray-700 transition-colors"
              >
                {q}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Input */}
      <div className="px-8 py-4 border-t border-gray-800 bg-gray-950">
        <form
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
          className="flex gap-3"
        >
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask about your team's health, risks, conflicts…"
            disabled={loading}
            className="flex-1 bg-gray-800 border border-gray-700 rounded-xl px-4 py-2.5 text-sm text-white placeholder-gray-500 focus:outline-none focus:border-cogni-500 transition-colors"
          />
          <button
            type="submit"
            disabled={loading || !input.trim()}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}
