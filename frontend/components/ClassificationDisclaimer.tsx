"use client";

import { useState } from "react";
import { CLASSIFICATION_DISCLAIMER } from "@/lib/types";

/**
 * Persistent scope/IFF disclaimer (v3 add-on Sections 0, 2.1, 4).
 * Rendered anywhere a classification label or priority score is shown.
 *
 * `variant="icon"`   -> small "i" info icon with a hover/click tooltip.
 * `variant="footer"` -> a fixed, always-visible footer strip.
 */
export default function ClassificationDisclaimer({
  variant = "icon",
}: {
  variant?: "icon" | "footer";
}) {
  const [open, setOpen] = useState(false);

  if (variant === "footer") {
    return (
      <div className="pointer-events-none fixed inset-x-0 bottom-0 z-40 flex justify-center px-3 pb-1">
        <p className="pointer-events-auto max-w-5xl rounded-t-md border border-white/10 border-b-0 bg-base-900/90 px-3 py-1 text-center font-mono text-[9px] leading-tight text-slate-400 backdrop-blur">
          <span className="text-amber">⚠ Illustrative behaviour patterns only.</span>{" "}
          {CLASSIFICATION_DISCLAIMER}
        </p>
      </div>
    );
  }

  return (
    <span className="relative inline-flex">
      <button
        type="button"
        aria-label="About classification labels"
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onClick={() => setOpen((v) => !v)}
        className="flex h-4 w-4 items-center justify-center rounded-full border border-slate-500 text-[9px] font-bold text-slate-400 hover:border-cyan hover:text-cyan"
      >
        i
      </button>
      {open && (
        <span className="absolute right-0 top-5 z-50 w-72 rounded-md border border-white/15 bg-base-900/95 p-2 text-left font-mono text-[9px] leading-snug text-slate-300 shadow-xl backdrop-blur">
          {CLASSIFICATION_DISCLAIMER}
        </span>
      )}
    </span>
  );
}
