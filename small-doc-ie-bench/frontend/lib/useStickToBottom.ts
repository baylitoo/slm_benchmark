"use client";

import { useCallback, useEffect, useRef } from "react";

/** How far from the bottom counts as "caught up" when deciding to resume. */
const NEAR_BOTTOM_PX = 64;

/**
 * Follow a growing transcript, unless the reader has scrolled away from it.
 *
 * **Why this does not decide from the scroll position alone.** A `scroll`
 * event is dispatched asynchronously, and during generation the content
 * changes every few milliseconds. So this interleaving is not merely possible,
 * it is the normal case:
 *
 * 1. the reader drags up — the browser scrolls, and queues a `scroll` event
 * 2. a token arrives before that event is delivered; the effect runs, still
 *    believes it is following, and jumps back to the bottom
 * 3. the queued event finally arrives and measures the position — which is
 *    the bottom again, because of step 2 — and concludes nobody moved
 *
 * The reader's scroll succeeded and was overwritten, and the evidence of it
 * was destroyed before anything could read it. No amount of care in the
 * measurement fixes that: by then it is measuring our own write.
 *
 * **So intent comes from the input, not from the position.** A wheel, a touch
 * drag or a navigation key is handled synchronously with the gesture, before
 * any effect can respond to it, which is what lets it beat the stream.
 * `scroll` is then used only to notice that the reader has *caught up* again,
 * where being late costs nothing.
 *
 * The rule the reader experiences is the usual one: scrolling up means "let me
 * read", returning to the bottom means "carry on", and sending means "show me
 * what happens next".
 */
export function useStickToBottom<T extends HTMLElement>(deps: readonly unknown[]) {
  const containerRef = useRef<T>(null);
  // Starts pinned: an empty transcript is at its own bottom, and the first
  // message should land in view without a scroll.
  const pinned = useRef(true);
  const touchY = useRef<number | null>(null);

  const toBottom = useCallback(() => {
    const el = containerRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  /** Only ever RE-pins. Unpinning is the input handlers' job, because this
   * one cannot tell the reader's scroll from the one we just performed. */
  const onScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight <= NEAR_BOTTOM_PX) {
      pinned.current = true;
    }
  }, []);

  /** Scrolling up is the gesture that means "stop following". Scrolling down
   * is left to `onScroll`, so reaching the bottom resumes. */
  const onWheel = useCallback((event: { deltaY: number }) => {
    if (event.deltaY < 0) pinned.current = false;
  }, []);

  const onTouchStart = useCallback((event: { touches: ArrayLike<{ clientY: number }> }) => {
    touchY.current = event.touches[0]?.clientY ?? null;
  }, []);

  /** A finger moving DOWN the screen scrolls the content UP. */
  const onTouchMove = useCallback((event: { touches: ArrayLike<{ clientY: number }> }) => {
    const start = touchY.current;
    const now = event.touches[0]?.clientY;
    if (start == null || now == null) return;
    if (now > start) pinned.current = false;
    touchY.current = now;
  }, []);

  const onKeyDown = useCallback((event: { key: string }) => {
    if (event.key === "ArrowUp" || event.key === "PageUp" || event.key === "Home") {
      pinned.current = false;
    }
  }, []);

  useEffect(() => {
    if (!pinned.current) return;
    toBottom();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  /** Re-pin regardless of where the reader was.
   *
   * Sending is an explicit "show me what happens next", so it overrides a
   * scroll position left from reading something earlier. */
  const pin = useCallback(() => {
    pinned.current = true;
    toBottom();
  }, [toBottom]);

  return {
    containerRef,
    /** Spread onto the scroll container. */
    handlers: { onScroll, onWheel, onTouchStart, onTouchMove, onKeyDown },
    pin,
  };
}
