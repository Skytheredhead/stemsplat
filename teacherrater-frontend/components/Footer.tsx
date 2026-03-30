"use client";

import { usePathname } from "next/navigation";

export function Footer() {
  const pathname = usePathname();
  const isModeratorOnly = process.env.NEXT_PUBLIC_MODERATOR_ONLY === "true";

  if (pathname.startsWith("/moderators") || isModeratorOnly) {
    return null;
  }

  return (
    <footer className="mt-auto border-t border-white/40 py-4 text-center text-xs text-slate-600 glass-panel">
      <p>
        Copyright 2026 - Not affiliated with anyone, any district, any school. Complaints/Suggestions:{" "}
        <a
          className="font-medium text-theme-primary hover:underline"
          href="https://forms.gle/yDoDPEqjcdzkoWyP6"
          target="_blank"
          rel="noreferrer"
        >
          https://forms.gle/yDoDPEqjcdzkoWyP6
        </a>
      </p>
    </footer>
  );
}
