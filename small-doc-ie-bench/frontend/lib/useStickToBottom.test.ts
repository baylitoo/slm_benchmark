import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useStickToBottom } from "@/lib/useStickToBottom";

/**
 * A transcript follows a streaming answer, until the reader scrolls away.
 *
 * The bug this replaces: the old effect scrolled to the bottom whenever the
 * messages changed, and the messages change on every streamed token — so
 * dragging up was undone milliseconds later and the view looked frozen at the
 * bottom for as long as generation lasted.
 */

/** A scroll container whose geometry the test controls. */
function container(
  { scrollHeight = 1000, clientHeight = 200, scrollTop = 800 } = {},
): HTMLDivElement {
  const el = document.createElement("div");
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true });
  Object.defineProperty(el, "clientHeight", { value: clientHeight, writable: true });
  el.scrollTop = scrollTop;
  return el;
}

/** scrollHeight is read-only in the DOM types; a growing transcript is the
 * whole point here, so it is redefined rather than assigned. */
function grow(el: HTMLDivElement, scrollHeight: number): void {
  Object.defineProperty(el, "scrollHeight", { value: scrollHeight, configurable: true });
}

function mount(el: HTMLDivElement, deps: readonly unknown[] = []) {
  return renderHook(({ d }: { d: readonly unknown[] }) => {
    const api = useStickToBottom<HTMLDivElement>(d);
    (api.containerRef as { current: HTMLDivElement | null }).current = el;
    return api;
  }, { initialProps: { d: deps } });
}

describe("useStickToBottom", () => {
  it("follows new content while the reader is at the bottom", () => {
    const el = container({ scrollTop: 800 }); // 1000 - 800 - 200 = 0 from bottom
    const { result, rerender } = mount(el, [1]);

    result.current.onScroll();
    grow(el, 1400);
    rerender({ d: [2] });

    expect(el.scrollTop).toBe(1400);
  });

  it("stops following the moment the reader scrolls up", () => {
    // The whole bug: this is what happens on every token while you drag.
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    el.scrollTop = 100;
    result.current.onScroll();
    grow(el, 1400);
    rerender({ d: [2] });

    expect(el.scrollTop).toBe(100);
  });

  it("treats a couple of lines from the bottom as still following", () => {
    // A streaming answer grows the content between a scroll event and the next
    // frame, so an exact-bottom test would unpin a reader who never moved.
    const el = container({ scrollTop: 760 }); // 40px from the bottom
    const { result, rerender } = mount(el, [1]);

    result.current.onScroll();
    grow(el, 1400);
    rerender({ d: [2] });

    expect(el.scrollTop).toBe(1400);
  });

  it("resumes following when the reader comes back down", () => {
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    el.scrollTop = 100;
    result.current.onScroll();
    rerender({ d: [2] });
    expect(el.scrollTop).toBe(100);

    el.scrollTop = 800;
    result.current.onScroll();
    grow(el, 1400);
    rerender({ d: [3] });

    expect(el.scrollTop).toBe(1400);
  });

  it("re-pins on send, wherever the reader was", () => {
    // Sending is an explicit "show me what happens next". Without this, someone
    // who scrolled up to check an older answer sends a new one and sees nothing.
    const el = container({ scrollTop: 100 });
    const { result } = mount(el, [1]);

    result.current.onScroll();
    result.current.pin();

    expect(el.scrollTop).toBe(1000);
  });

  it("starts pinned, so the first message lands in view", () => {
    const el = container({ scrollTop: 0, scrollHeight: 0, clientHeight: 200 });
    const { rerender } = mount(el, [1]);

    grow(el, 600);
    rerender({ d: [2] });

    expect(el.scrollTop).toBe(600);
  });
});
