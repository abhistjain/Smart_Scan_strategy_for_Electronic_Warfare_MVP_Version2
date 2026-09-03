"use client";

/**
 * Official DRDO emblem. Served from /public so both the setup screen and the
 * dashboard header can reuse the same mark without bundling a huge JPEG twice.
 */
export default function DrdoLogo({
  size = 40,
  className = "",
}: {
  size?: number;
  className?: string;
}) {
  return (
    <img
      src="/drdo-logo.jpg"
      alt="Defence Research and Development Organisation"
      width={size}
      height={size}
      className={`shrink-0 rounded-full bg-white object-cover ring-1 ring-white/25 ${className}`}
      style={{ width: size, height: size }}
    />
  );
}
