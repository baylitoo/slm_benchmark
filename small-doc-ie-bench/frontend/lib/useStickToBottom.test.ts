import { renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { useStickToBottom } from "@/lib/useStickToBottom";

/**
 * A transcript follows a streaming answer, until the reader scrolls away.
 *
 * The first attempt decided from the scroll position and shipped broken,
 * because position arrives via a `scroll` event and that event is
 * asynchronous: during generation a token lands between the reader's drag and
 * the event that would have reported it, the effect jumps back to the bottom,
 * and the event then measures our own write. These tests model that ordering
 * rather than the tidy one — which is why the first version passed its tests
 * and still lost every fight with the reader.
 */

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

/** A token arriving: the content grew, and the effect gets to run. */
function token(
  el: HTMLDivElement,
  rerender: (p: { d: readonly unknown[] }) => void,
  height: number,
  n: number,
) {
  grow(el, height);
  rerender({ d: [n] });
}

describe("useStickToBottom", () => {
  it("follows new content while the reader has not moved", () => {
    const el = container({ scrollTop: 800 });
    const { rerender } = mount(el, [1]);

    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(1400);
  });

  it("stops following the moment the wheel turns, before any scroll event", () => {
    // THE BUG. The reader's drag is reported asynchronously, so a token can
    // land first. Reading intent from the position would jump to the bottom
    // here, and the queued event would then measure that jump and conclude
    // nobody had moved.
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: -120 });
    el.scrollTop = 100;
    // Deliberately no onScroll: this is the window the first version lost.
    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(100);
  });

  it("stays away for every token that follows, not only the first", () => {
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: -120 });
    el.scrollTop = 100;
    token(el, rerender, 1400, 2);
    token(el, rerender, 1800, 3);
    token(el, rerender, 2200, 4);

    expect(el.scrollTop).toBe(100);
  });

  it("keeps following when the wheel turns downward", () => {
    // Scrolling toward the bottom is not a request to stop.
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: 120 });
    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(1400);
  });

  it("resumes once the reader reaches the bottom again", () => {
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: -120 });
    el.scrollTop = 100;
    token(el, rerender, 1400, 2);
    expect(el.scrollTop).toBe(100);

    el.scrollTop = 1200; // 1400 - 1200 - 200 = 0 from the bottom
    result.current.handlers.onScroll();
    token(el, rerender, 1800, 3);

    expect(el.scrollTop).toBe(1800);
  });

  it("treats a couple of lines short of the bottom as caught up", () => {
    // Content grows between the scroll and the next frame, so an exact test
    // would refuse to resume for a reader who did scroll all the way down.
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: -120 });
    el.scrollTop = 100;
    token(el, rerender, 1400, 2);

    el.scrollTop = 1160; // 40px short
    result.current.handlers.onScroll();
    token(el, rerender, 1800, 3);

    expect(el.scrollTop).toBe(1800);
  });

  it("a scroll event alone never stops it following", () => {
    // onScroll cannot tell the reader's scroll from the one the effect just
    // performed, so it is not allowed to unpin on what it sees.
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    el.scrollTop = 100;
    result.current.handlers.onScroll();
    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(1400);
  });

  it("stops for a navigation key", () => {
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onKeyDown({ key: "PageUp" });
    el.scrollTop = 100;
    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(100);
  });

  it("stops for a finger dragging the content up", () => {
    const el = container({ scrollTop: 800 });
    const { result, rerender } = mount(el, [1]);

    result.current.handlers.onTouchStart({ touches: [{ clientY: 100 }] });
    result.current.handlers.onTouchMove({ touches: [{ clientY: 180 }] });
    el.scrollTop = 100;
    token(el, rerender, 1400, 2);

    expect(el.scrollTop).toBe(100);
  });

  it("re-pins on send, wherever the reader was", () => {
    const el = container({ scrollTop: 100 });
    const { result } = mount(el, [1]);

    result.current.handlers.onWheel({ deltaY: -120 });
    result.current.pin();

    expect(el.scrollTop).toBe(1000);
  });

  it("starts pinned, so the first message lands in view", () => {
    const el = container({ scrollTop: 0, scrollHeight: 0, clientHeight: 200 });
    const { rerender } = mount(el, [1]);

    token(el, rerender, 600, 2);

    expect(el.scrollTop).toBe(600);
  });
});
