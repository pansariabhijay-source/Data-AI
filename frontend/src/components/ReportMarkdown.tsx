"use client";

import React from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

/**
 * Premium markdown renderer for Axiom pipeline reports.
 *
 * Uses react-markdown + remark-gfm (GitHub-Flavored Markdown) so tables,
 * task lists, strikethrough, autolinks, etc. all render correctly — replacing
 * the previous hand-rolled parser that silently dropped edge cases. Every
 * element is themed to match the dark Axiom UI for a professional, report-grade
 * look (rounded zebra-striped tables, accent-barred section headings, styled
 * code, callout blockquotes).
 */

function isNumeric(text: string): boolean {
  // Right-align numeric-looking cells (counts, metrics, currency, %).
  return /^[$£€]?-?[\d,]+(\.\d+)?%?$/.test(text.trim());
}

function cellText(children: React.ReactNode): string {
  if (typeof children === "string") return children;
  if (Array.isArray(children)) return children.map(cellText).join("");
  return "";
}

export default function ReportMarkdown({ content }: { content: string }) {
  return (
    <div className="report-md text-[13.5px] leading-[1.85] text-text-secondary">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          h1: ({ children }) => (
            <h1 className="text-[26px] font-bold tracking-tight gradient-text mt-2 mb-5">
              {children}
            </h1>
          ),
          h2: ({ children }) => (
            <h2 className="flex items-center gap-3 text-[18px] font-semibold text-text-primary mt-11 mb-4 pt-5 border-t border-glass-border/40">
              <span className="w-1 h-5 rounded-full bg-accent inline-block shrink-0" />
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-[15px] font-semibold text-text-secondary mt-7 mb-2.5">
              {children}
            </h3>
          ),
          p: ({ children }) => (
            <p className="my-3 leading-[1.85] text-text-secondary">{children}</p>
          ),
          a: ({ children, href }) => (
            <a
              href={href}
              target="_blank"
              rel="noopener noreferrer"
              className="text-accent underline decoration-accent/30 underline-offset-2 hover:decoration-accent transition-colors"
            >
              {children}
            </a>
          ),
          strong: ({ children }) => (
            <strong className="font-semibold text-text-primary">{children}</strong>
          ),
          em: ({ children }) => <em className="italic text-text-secondary">{children}</em>,
          code: ({ className, children }) => {
            const isBlock = (className || "").includes("language-");
            if (isBlock) {
              return (
                <code className="block font-mono text-[12.5px] text-text-secondary">
                  {children}
                </code>
              );
            }
            return (
              <code className="font-mono text-[12px] text-accent bg-accent/[0.08] border border-accent/15 px-1.5 py-0.5 rounded-md">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="my-4 p-4 rounded-xl bg-void/60 border border-glass-border overflow-x-auto">
              {children}
            </pre>
          ),
          ul: ({ children }) => <ul className="my-4 space-y-2 pl-1">{children}</ul>,
          ol: ({ children }) => (
            <ol className="my-4 space-y-2 pl-5 list-decimal marker:text-text-muted">
              {children}
            </ol>
          ),
          li: ({ children }) => (
            <li className="flex items-start gap-2.5 text-text-secondary leading-relaxed [ol_&]:list-item [ol_&]:pl-1">
              <span className="mt-2 h-1.5 w-1.5 rounded-full bg-accent/70 shrink-0 [ol_&]:hidden" />
              <span className="flex-1">{children}</span>
            </li>
          ),
          blockquote: ({ children }) => (
            <blockquote className="my-5 pl-4 pr-4 py-3 border-l-2 border-accent/50 bg-accent/[0.04] rounded-r-lg text-text-secondary italic">
              {children}
            </blockquote>
          ),
          hr: () => <hr className="my-9 border-glass-border/50" />,
          table: ({ children }) => (
            <div className="my-6 overflow-x-auto rounded-xl border border-glass-border shadow-sm">
              <table className="w-full border-collapse text-[13px]">{children}</table>
            </div>
          ),
          thead: ({ children }) => (
            <thead className="bg-white/[0.04]">{children}</thead>
          ),
          th: ({ children }) => {
            const align = isNumeric(cellText(children)) ? "text-right" : "text-left";
            return (
              <th
                className={`${align} py-3 px-5 text-[10px] font-bold uppercase tracking-[1.4px] text-text-muted border-b border-glass-border whitespace-nowrap`}
              >
                {children}
              </th>
            );
          },
          tbody: ({ children }) => (
            <tbody className="[&>tr:nth-child(even)]:bg-white/[0.015]">{children}</tbody>
          ),
          tr: ({ children }) => (
            <tr className="border-b border-glass-border/30 last:border-0 hover:bg-accent/[0.04] transition-colors">
              {children}
            </tr>
          ),
          td: ({ children }) => {
            const text = cellText(children);
            const numeric = isNumeric(text);
            return (
              <td
                className={`py-2.5 px-5 align-top ${
                  numeric
                    ? "text-right font-mono text-text-primary tabular-nums"
                    : "text-left text-text-secondary"
                }`}
              >
                {children}
              </td>
            );
          },
        }}
      >
        {content}
      </ReactMarkdown>
    </div>
  );
}
