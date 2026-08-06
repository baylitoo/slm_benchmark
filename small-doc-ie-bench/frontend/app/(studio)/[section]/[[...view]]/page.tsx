import { notFound } from "next/navigation";
import { SECTIONS } from "@/components/shell/nav";

/**
 * "/{section}" and "/{section}/{view}" — validates the section so an unknown
 * URL is a real 404 (app/not-found.tsx) instead of silently rendering the
 * default section. The shell itself is rendered by the (studio) layout and
 * derives section + view from the pathname.
 */
export default async function SectionPage({
  params,
}: {
  params: Promise<{ section: string }>;
}) {
  const { section } = await params;
  if (!(SECTIONS as readonly string[]).includes(section)) notFound();
  return null;
}
