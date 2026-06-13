import type { ReactNode } from "react";

/**
 * Solo route wrapper — fills the available viewport with a single panel.
 * Used for the per-stream routes (/selfplay, /training, etc.) where the
 * user wants a focused, larger view of one thing.
 */
export function SoloView({ children }: { children: ReactNode }) {
  return <div className="solo-view">{children}</div>;
}
