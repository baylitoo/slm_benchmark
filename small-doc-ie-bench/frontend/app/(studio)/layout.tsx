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
 * the build fails with "should be wrapped in a suspense boundary". No
 * fallback UI is needed: this is a fully client-rendered SPA shell, not a
 * page doing SSR/static generation, so the boundary is never actually
 * shown mid-render.
 */
export default function StudioLayout({ children }: { children: React.ReactNode }) {
  return (
    <Suspense>
      <AppShell />
      {children}
    </Suspense>
  );
}
