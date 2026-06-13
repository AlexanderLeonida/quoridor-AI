import { useEffect, useState } from "react";

/**
 * Fetch the list of available streams from the backend exactly once on
 * mount.  Used by both the combined dashboard and the solo views to know
 * which panels to render vs. show a "not available" message.
 */
export function useStreams(): {
  streams: string[];
  loaded: boolean;
} {
  const [streams, setStreams] = useState<string[]>([]);
  const [loaded, setLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    fetch("/api/streams")
      .then((r) => r.json())
      .then((j) => {
        if (!cancelled) {
          setStreams(j.streams ?? []);
          setLoaded(true);
        }
      })
      .catch(() => {
        if (!cancelled) setLoaded(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { streams, loaded };
}
