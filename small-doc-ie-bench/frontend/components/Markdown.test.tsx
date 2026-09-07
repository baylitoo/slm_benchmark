import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Markdown } from "@/components/Markdown";

describe("Markdown", () => {
  it("renders emphasis, lists, and inline code", () => {
    render(<Markdown text={"**total** is `42`\n\n- a\n- b"} />);
    const bold = screen.getByText("total");
    expect(bold.tagName).toBe("STRONG");
    expect(screen.getByText("42").tagName).toBe("CODE");
    expect(screen.getAllByRole("listitem")).toHaveLength(2);
  });

  it("renders GFM tables", () => {
    render(<Markdown text={"| qty | price |\n| --- | --- |\n| 3 | 129.99 |"} />);
    expect(screen.getByRole("table")).toBeInTheDocument();
    expect(screen.getByText("129.99").tagName).toBe("TD");
  });

  it("renders fenced code blocks in a pre", () => {
    render(<Markdown text={'```json\n{"total": 42}\n```'} />);
    const code = screen.getByText(/"total": 42/);
    expect(code.closest("pre")).not.toBeNull();
  });

  it("escapes raw HTML from model output", () => {
    render(<Markdown text={'<img src=x onerror="x()"> hello'} />);
    // react-markdown drops/escapes raw HTML nodes -- no <img> element mounts.
    expect(document.querySelector("img")).toBeNull();
    expect(screen.getByText(/hello/)).toBeInTheDocument();
  });

  it("opens links in a new tab with rel protection", () => {
    render(<Markdown text={"[doc](https://example.com)"} />);
    const link = screen.getByRole("link", { name: "doc" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", expect.stringContaining("noopener"));
  });
});
