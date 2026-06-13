import { useEffect, useRef, useState } from "react";

type Status = "connecting" | "open" | "closed" | "error";

/**
 * Subscribe to one of the backend's WebSocket streams.
 *
 * Reconnects with exponential backoff if the socket drops.
 */
export function useWebSocketStream<T>(name: string): {
  data: T | null;
  status: Status;
} {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<Status>("connecting");
  const reconnectRef = useRef(0);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    let cancelled = false;

    const connect = () => {
      if (cancelled) return;
      const proto = window.location.protocol === "https:" ? "wss" : "ws";
      const url = `${proto}://${window.location.host}/ws/${name}`;
      setStatus("connecting");
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        reconnectRef.current = 0;
        setStatus("open");
      };
      ws.onmessage = (ev) => {
        try {
          setData(JSON.parse(ev.data) as T);
        } catch {
          // ignore malformed payloads
        }
      };
      ws.onerror = () => setStatus("error");
      ws.onclose = () => {
        setStatus("closed");
        if (cancelled) return;
        const delay = Math.min(
          10_000,
          500 * 2 ** Math.min(reconnectRef.current, 5),
        );
        reconnectRef.current += 1;
        setTimeout(connect, delay);
      };
    };

    connect();
    return () => {
      cancelled = true;
      wsRef.current?.close();
    };
  }, [name]);

  return { data, status };
}
