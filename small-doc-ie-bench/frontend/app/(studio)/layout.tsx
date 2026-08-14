import { Suspense } from "react";
import { AppShell } from "@/components/AppShell";

/**
 * The Studio shell lives in this LAYOUT, not in the pages: layouts persist
 * across route navigation, so every section (and its in-flight extraction /
 * deploy / benchmark work) survives URL changes exactly like the old
 * setActive() model — while the URL becomes the navigation source of truth
 * (deep links, back button, refresh). The route pages below render nothing;
 * AppShell reads the pathname itself.
 *
 * AppShell also reads useSearchParams() (for deep-link filters, e.g.
 * Observability's activity tile jumping into Deployments pre-filtered) --
 * Next.js requires that hook's caller to sit inside a Suspense boundary or
 * the build fails with "should be wrapped in a suspense boundary".
 *
 * No fallback UI here, but that's not free everywhere: the dynamic routes
 * this feature actually targets (e.g. /deploy/deployments?q=...) are
 * server-rendered per request with no CSR bailout, so the boundary is
 * never visibly hit there. The bare "/" route, however, IS static-
 * prerendered (confirmed via `next build`'s own output) and DOES bail to
 * client-side rendering for the whole shell -- a hard load/refresh of "/"
 * itself briefly renders nothing (React's default empty Suspense
 * fallback) until hydration. Root landing on "/" isn't a normal user path
 * (every real navigation carries a section), so left unaddressed for now;
 * revisit with a real skeleton fallback if that blank-flash starts
 * mattering in practice.
 */
export default function StudioLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense>
      <AppShell />
      {children}
    </Suspense>
  );
}
