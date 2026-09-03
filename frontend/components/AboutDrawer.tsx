"use client";

import { AnimatePresence, motion } from "framer-motion";

// Plain-language "how it works" slide-over for judges (spec 9.4).
export default function AboutDrawer({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-40 bg-black/60"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            className="glass-strong fixed right-0 top-0 z-50 h-full w-full max-w-md overflow-y-auto p-6"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 26, stiffness: 240 }}
          >
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-lg font-semibold text-slate-100">How it works</h2>
              <button
                onClick={onClose}
                className="rounded bg-white/5 px-2 py-1 font-mono text-xs text-slate-400 hover:text-cyan"
              >
                ✕ close
              </button>
            </div>

            <div className="space-y-4 text-sm leading-relaxed text-slate-300">
              <Section title="The problem" color="#22D3EE">
                An Electronic Support (ES) receiver can only listen to a few
                frequency bands at once, but must find hostile emitters across a
                huge spectrum. Interception is a 2-D search: be at the right
                frequency at the right time — with no prior intelligence on the
                emitters.
              </Section>

              <Section title="1 · Bayesian belief filter" color="#22D3EE">
                For every band we keep a probability that it is currently active,
                updated by Bayes&apos; rule whenever we scan it and propagated by a
                Markov model when we don&apos;t. This lets us reason about bands we
                are not even looking at.
              </Section>

              <Section title="2 · Thompson sampling" color="#F5A623">
                We do not know each band&apos;s transition probabilities, so we learn
                them online with Beta posteriors and <em>sample</em> from them each
                tick. Sampling is what drives exploration — no hand-tuned
                epsilon-greedy.
              </Section>

              <Section title="3 · Whittle index" color="#A78BFA">
                Each band gets a scan-priority score from the closed-form Whittle
                index for restless bandits (Liu &amp; Zhao, 2010). We scan the
                Top-M scoring bands. This is provably near-optimal and balances
                exploiting likely-active bands against revisiting stale ones.
              </Section>

              <Section title="4 · Intercept-ahead (periodicity)" color="#F472B6">
                A Lomb-Scargle periodogram plus a von Mises phase fit detect
                periodic (rotating-antenna) emitters and predict their next active
                window, boosting that band&apos;s priority just before it lights up.
              </Section>

              <Section title="Why it wins" color="#34D399">
                Against a fixed sequential sweep, random, and greedy baselines run
                on the identical environment, the smart scheduler intercepts far
                more, faster — typically ~2–3× the interception rate of the
                open-loop sweep.
              </Section>

              <Section title="Real data" color="#A78BFA">
                &quot;Real · TSRD&quot; mode replays an occupancy grid built offline
                from Stare-mode pulse trains of the Turing Synthetic Radar Dataset,
                so the scheduler is validated on realistic, non-hand-crafted
                emitter behaviour.
              </Section>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function Section({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  return (
    <div>
      <h3 className="mb-1 font-mono text-xs uppercase tracking-widest" style={{ color }}>
        {title}
      </h3>
      <p className="text-slate-400">{children}</p>
    </div>
  );
}
