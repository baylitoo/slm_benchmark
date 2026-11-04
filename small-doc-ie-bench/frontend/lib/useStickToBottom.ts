"use client";

import { useCallback, useEffect, useRef } from "react";

/** How far from the bottom still counts as "following along".
 *
 * Not zero: a streaming answer grows the content between a scroll event and
 * the next frame, so an exact-bottom test unpins a reader who never moved.
 * Not large either, or someone reading a screen above the fold gets yanked
 * down anyway. A couple of lines of text. */
const NEAR_BOTTOM_PX = 64;

/**
 * Follow a growing transcript, unless the reader has scrolled away from it.
 *
 * The naive version — scroll to the bottom whenever the messages change —
 * fights the user during generation, because the messages change on every
 * streamed token. Each delta re-issues the scroll, so dragging up is undone
 * a few milliseconds later and the view appears stuck at the bottom.
 *
 * `behavior: "smooth"` makes that worse rather than better: a smooth scroll
 * is still animating when the next token restarts it, so the viewport is
 * never not being scrolled. This assigns `scrollTop` directly, which is
 * instant and cannot queue.
 *
 * The rule is the one every chat client converges on: stay pinned while the
 * reader is at the bottom, stop the moment they leave, resume when they come
 * back. Scrolling up is how you say "let me read"; scrolling down is how you
 * say "carry on".
 */
export function useStickToBottom<T extends HTMLElement>(deps: readonly unknown[]) {
  const containerRef = useRef<T>(null);
  // Starts pinned: an empty transcript is at its own bottom, and the first
  // message should land in view without a scroll.
  const pinned = useRef(true);

  const onScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    pinned.current = el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_PX;
  }, []);

  useEffect(() => {
    const el = containerRef.current;
    if (!el || !pinned.current) return;
    el.scrollTop = el.scrollHeight;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  /** Re-pin regardless of where the reader was.
   *
   * Sending is an explicit "show me what happens next", so it overrides a
   * scroll position from reading something earlier. Without this, a reader who
   * scrolled up to check an old answer sends a new one and watches nothing. */
  const pin = useCallback(() => {
    pinned.current = true;
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  return { containerRef, onScroll, pin };
}
