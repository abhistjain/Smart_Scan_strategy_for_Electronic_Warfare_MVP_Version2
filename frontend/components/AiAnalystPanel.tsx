"use client";

import { useEffect, useRef, useState } from "react";
import { AnimatePresence, motion } from "framer-motion";
import { useStore } from "@/lib/store";
import { aiChat, aiStatus, setPriorityWeights } from "@/lib/api";
import type { AIStatus } from "@/lib/types";
import ClassificationDisclaimer from "./ClassificationDisclaimer";
import FormattedAIText from "./FormattedAIText";

interface ChatMsg {
  role: "user" | "assistant";
  text: string;
  unavailable?: boolean;
}

const SUGGESTIONS = [
  "Which bands look most concerning right now and why?",
  "Explain why the Smart Scheduler is outperforming the sequential sweep.",
  "Which missile should intercept the top band?", // out-of-scope probe
];

function PriorityWeights({ scenarioId }: { scenarioId: string }) {
  const [w, setW] = useState({ w_belief: 0.5, w_conf: 0.2, w_urgency: 0.3 });
  const timer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const update = (key: keyof typeof w, value: number) => {
    const next = { ...w, [key]: value };
    setW(next);
    if (timer.current) clearTimeout(timer.current);
    timer.current = setTimeout(() => setPriorityWeights(scenarioId, next).catch(() => {}), 250);
  };

  const rows: [keyof typeof w, string, string][] = [
    ["w_belief", "belief", "#22D3EE"],
    ["w_conf", "confidence", "#34D399"],
    ["w_urgency", "urgency", "#F5A623"],
  ];

  return (
    <div className="border-b border-white/10 p-3">
      <div className="mb-2 flex items-center justify-between">
        <span className="font-mono text-[9px] uppercase tracking-widest text-slate-500">
          Priority weights (attention sort only)
        </span>
        <ClassificationDisclaimer variant="icon" />
      </div>
      <div className="space-y-2">
        {rows.map(([key, label, color]) => (
          <div key={key} className="flex items-center gap-2">
            <span className="w-20 shrink-0 font-mono text-[10px]" style={{ color }}>
              {label}
            </span>
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={w[key]}
              onChange={(e) => update(key, parseFloat(e.target.value))}
              className="flex-1 accent-cyan"
            />
            <span className="w-8 text-right font-mono text-[10px] text-slate-300">
              {w[key].toFixed(2)}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function AiAnalystPanel({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const scenarioId = useStore((s) => s.scenarioId);
  const [status, setStatus] = useState<AIStatus | null>(null);
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (open) aiStatus().then(setStatus).catch(() => setStatus(null));
  }, [open]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, loading]);

  const send = async (question: string) => {
    if (!scenarioId || !question.trim() || loading) return;
    setInput("");
    setMessages((m) => [...m, { role: "user", text: question }]);
    setLoading(true);
    try {
      const res = await aiChat(scenarioId, question);
      setMessages((m) => [
        ...m,
        { role: "assistant", text: res.text, unavailable: !res.available },
      ]);
    } catch {
      setMessages((m) => [
        ...m,
        { role: "assistant", text: "AI narration unavailable", unavailable: true },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            className="glass fixed right-0 top-0 z-50 flex h-full w-full max-w-md flex-col border-l border-white/10"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 26, stiffness: 220 }}
          >
            {/* Header */}
            <div className="flex items-center justify-between border-b border-white/10 p-3">
              <div>
                <h3 className="font-mono text-sm text-slate-200">AI Analyst</h3>
                <div className="flex items-center gap-1.5 font-mono text-[9px]">
                  <span
                    className={`h-1.5 w-1.5 rounded-full ${status?.available ? "bg-emerald-400" : "bg-slate-500"}`}
                  />
                  <span className="text-slate-500">
                    {status?.available
                      ? `online · ${status.model}`
                      : `offline · ${status?.reason ?? "no API key"}`}
                  </span>
                </div>
              </div>
              <button
                onClick={onClose}
                className="rounded px-2 py-1 font-mono text-xs text-slate-400 hover:bg-white/10"
              >
                ✕
              </button>
            </div>

            {scenarioId && <PriorityWeights scenarioId={scenarioId} />}

            {/* Messages */}
            <div ref={scrollRef} className="flex-1 space-y-2 overflow-y-auto p-3">
              {messages.length === 0 && (
                <div className="space-y-2">
                  <p className="font-mono text-[10px] text-slate-500">
                    Ask about signal behaviour patterns, band priorities, or scheduler
                    performance. Weapon/engagement questions are out of scope.
                  </p>
                  {SUGGESTIONS.map((s) => (
                    <button
                      key={s}
                      onClick={() => send(s)}
                      className="block w-full rounded border border-white/10 bg-white/5 px-2 py-1.5 text-left font-mono text-[10px] text-slate-300 hover:border-cyan/40 hover:bg-cyan/5"
                    >
                      {s}
                    </button>
                  ))}
                </div>
              )}
              {messages.map((m, i) => (
                <div
                  key={i}
                  className={`max-w-[90%] rounded-lg px-2.5 py-1.5 text-xs leading-relaxed ${
                    m.role === "user"
                      ? "ml-auto bg-cyan/15 text-cyan-100"
                      : "bg-white/5"
                  }`}
                >
                  {m.role === "user" ? (
                    m.text
                  ) : (
                    <FormattedAIText text={m.text} muted={m.unavailable} />
                  )}
                </div>
              ))}
              {loading && (
                <div className="max-w-[60%] rounded-lg bg-white/5 px-2.5 py-2">
                  <div className="flex gap-1">
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.3s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400 [animation-delay:-0.15s]" />
                    <span className="h-1.5 w-1.5 animate-bounce rounded-full bg-slate-400" />
                  </div>
                </div>
              )}
            </div>

            {/* Input */}
            <div className="border-t border-white/10 p-3">
              <div className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && send(input)}
                  placeholder="Ask the analyst…"
                  className="flex-1 rounded border border-white/10 bg-white/5 px-2 py-1.5 font-mono text-xs text-slate-200 placeholder:text-slate-600 focus:border-cyan/50 focus:outline-none"
                />
                <button
                  onClick={() => send(input)}
                  disabled={loading}
                  className="rounded bg-cyan/20 px-3 py-1.5 font-mono text-xs text-cyan hover:bg-cyan/30 disabled:opacity-40"
                >
                  Send
                </button>
              </div>
              <p className="mt-1.5 text-right font-mono text-[9px] text-slate-600">
                Powered by Claude · sensing &amp; classification only
              </p>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
