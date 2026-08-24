"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/cn";

// Markdown for model output (chat bubbles, vision answers). GFM enabled for
// the tables models emit on invoice questions; raw HTML stays escaped
// (react-markdown default), so model output can't inject markup.
export function Markdown({ text, className }: { text: string; className?: string }) {
  return (
    <div className={cn("space-y-2 break-words text-sm", className)}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          p: ({ children }) => <p className="whitespace-pre-wrap">{children}</p>,
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noreferrer noopener"
              className="underline decoration-dotted underline-offset-2"
            >
              {children}
            </a>
          ),
          ul: ({ children }) => <ul className="list-disc space-y-0.5 pl-5">{children}</ul>,
          ol: ({ children }) => <ol className="list-decimal space-y-0.5 pl-5">{children}</ol>,
          h1: ({ children }) => <p className="font-semibold">{children}</p>,
          h2: ({ children }) => <p className="font-semibold">{children}</p>,
          h3: ({ children }) => <p className="font-semibold">{children}</p>,
          code: ({ children, className: codeClass }) =>
            codeClass ? (
              // Block code (```lang fences get a language- class).
              <code className={cn("block", codeClass)}>{children}</code>
            ) : (
              <code className="rounded bg-muted px-1 py-0.5 font-mono text-xs">{children}</code>
            ),
          pre: ({ children }) => (
            <pre className="overflow-x-auto rounded-md bg-muted p-3 font-mono text-xs">
              {children}
            </pre>
          ),
          table: ({ children }) => (
            <div className="overflow-x-auto">
              <table className="w-full border-collapse text-xs">{children}</table>
            </div>
          ),
          th: ({ children }) => (
            <th className="border border-border bg-muted/50 px-2 py-1 text-left font-medium">
              {children}
            </th>
          ),
          td: ({ children }) => (
            <td className="border border-border px-2 py-1">{children}</td>
          ),
          blockquote: ({ children }) => (
            <blockquote className="border-l-2 border-border pl-3 text-muted-foreground">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="border-border" />,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
