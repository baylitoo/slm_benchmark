"use client";

import { useEffect, useState } from "react";
import { getRuntimes } from "./api";

export type Health = "checking" | "online" | "offline";

/**
 * Lightweight backend liveness probe. Hits the always-available
 * GET /v1/serving/runtimes endpoint on an interval (paused when hidden).
 */
export function useBackendHealth(intervalMs = 10000): Health {
  const [health, setHealth] = useState<Health>("checking");

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setInterval> | null = null;

    const ping = async () => {
      try {
        await getRuntimes();
        if (!cancelled) setHealth("online");
      } catch {
        if (!cancelled) setHealth("offline");
      }
    };

    const start = () => {
      if (timer != null) return;
      void ping();
      timer = setInterval(() => void ping(), intervalMs);
    };
    const stop = () => {
      if (timer != null) {
        clearInterval(timer);
        timer = null;
      }
    };
    const sync = () => {
      if (document.visibilityState === "visible") start();
      else stop();
    };

    sync();
    document.addEventListener("visibilitychange", sync);
    return () => {
      cancelled = true;
      document.removeEventListener("visibilitychange", sync);
      stop();
    };
  }, [intervalMs]);

  return health;
}
