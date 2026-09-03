"use client";

import { Fragment, useMemo } from "react";

/**
 * Minimal, dependency-free renderer for the short markdown-ish text the AI
 * Analyst returns (chat replies, band narration, end-of-run summary).
 *
 * It intentionally supports only what the model actually emits so we never ship
 * a heavy markdown engine to the client:
 *   - blank-line separated paragraphs
 *   - bullet lines starting with "-", "*" or "•"
 *   - inline **bold** and `code`
 *   - long " - " / " · " separated attribute runs are split onto their own
 *     bullet lines so a wall of dashes becomes a scannable list.
 */

type Block =
  | { type: "p"; text: string }
  | { type: "ul"; items: string[] };

const BULLET_RE = /^\s*[-*•]\s+(.*)$/;

// A paragraph that is really an inline attribute list, e.g.
//   "**B08** (priority 0.49) - Behaviour: Comms-Like - Confidence: 0.88 - Belief: 0.10 - This is ..."
// We split on " - " / " · " when there are enough segments to look like a list.
function maybeSplitInlineList(text: string): string[] | null {
  const parts = text
    .split(/\s+[-·]\s+/g)
    .map((s) => s.trim())
    .filter(Boolean);
  if (parts.length < 3) return null;
  // Keep the leading segment (often the band label + score) as the intro line,
  // and only treat the shorter "Key: value" style segments as bullets.
  const looksLikeAttrs = parts.slice(1).filter((p) => p.length <= 60).length >= 2;
  return looksLikeAttrs ? parts : null;
}

function parseBlocks(raw: string): Block[] {
  const lines = raw.replace(/\r\n/g, "\n").split("\n");
  const blocks: Block[] = [];
  let para: string[] = [];
  let list: string[] = [];

  const flushPara = () => {
    if (!para.length) return;
    const joined = para.join(" ").trim();
    const inline = maybeSplitInlineList(joined);
    if (inline) {
      blocks.push({ type: "p", text: inline[0] });
      blocks.push({ type: "ul", items: inline.slice(1) });
    } else {
      blocks.push({ type: "p", text: joined });
    }
    para = [];
  };
  const flushList = () => {
    if (!list.length) return;
    blocks.push({ type: "ul", items: list });
    list = [];
  };

  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") {
      flushPara();
      flushList();
      continue;
    }
    const m = trimmed.match(BULLET_RE);
    if (m) {
      flushPara();
      list.push(m[1]);
    } else {
      flushList();
      para.push(trimmed);
    }
  }
  flushPara();
  flushList();
  return blocks;
}

function renderInline(text: string, keyPrefix: string) {
  // Split on **bold** and `code`, keeping delimiters.
  const tokens = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g).filter(Boolean);
  return tokens.map((tok, i) => {
    if (tok.startsWith("**") && tok.endsWith("**")) {
      return (
        <strong key={`${keyPrefix}-${i}`} className="font-semibold text-slate-100">
          {tok.slice(2, -2)}
        </strong>
      );
    }
    if (tok.startsWith("`") && tok.endsWith("`")) {
      return (
        <code
          key={`${keyPrefix}-${i}`}
          className="rounded bg-white/10 px-1 py-[1px] font-mono text-[0.95em] text-cyan"
        >
          {tok.slice(1, -1)}
        </code>
      );
    }
    return <Fragment key={`${keyPrefix}-${i}`}>{tok}</Fragment>;
  });
}

export default function FormattedAIText({
  text,
  className = "",
  muted = false,
}: {
  text: string;
  className?: string;
  muted?: boolean;
}) {
  const blocks = useMemo(() => parseBlocks(text), [text]);
  const base = muted ? "italic text-slate-500" : "text-slate-200";

  return (
    <div className={`space-y-2 text-xs leading-relaxed ${base} ${className}`}>
      {blocks.map((b, i) =>
        b.type === "p" ? (
          <p key={i}>{renderInline(b.text, `p${i}`)}</p>
        ) : (
          <ul key={i} className="space-y-1 pl-1">
            {b.items.map((it, j) => (
              <li key={j} className="flex gap-2">
                <span className="mt-[2px] shrink-0 text-cyan/70">›</span>
                <span className="flex-1">{renderInline(it, `l${i}-${j}`)}</span>
              </li>
            ))}
          </ul>
        ),
      )}
    </div>
  );
}
